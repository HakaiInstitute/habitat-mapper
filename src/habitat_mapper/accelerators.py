"""Hardware accelerator discovery and ONNX Runtime execution provider selection."""

from __future__ import annotations

import contextlib
import ctypes
import io
import os
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from functools import cache
from importlib.metadata import distributions
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args

from loguru import logger

with warnings.catch_warnings():
    # onnxruntime-gpu registers its CUDA provider on import and warns when the CUDA libraries can't be loaded.
    # That's expected without an NVIDIA GPU, so CUDA support is checked explicitly in find_accelerators instead.
    warnings.filterwarnings("ignore", message="Failed to register bundled CUDA plugin EP")
    try:
        import onnxruntime as ort
    except ImportError as e:
        # ONNX Runtime needs the Visual C++ runtime, which a fresh Windows install doesn't have
        if sys.platform == "win32" and "DLL load failed" in str(e):
            raise ImportError(
                "ONNX Runtime could not be loaded. Install the Microsoft Visual C++ Redistributable from "
                "https://aka.ms/vs/17/release/vc_redist.x64.exe and try again."
            ) from e
        raise

from onnxruntime.capi.onnxruntime_pybind11_state import InvalidProtobuf, NoSuchFile

if TYPE_CHECKING:
    from collections.abc import Collection, Generator, Sequence

# 4=FATAL. Errors are raised as exceptions and reported through loguru, so ORT's own error log is redundant
ort.set_default_logger_severity(4)

Device = Literal["auto", "cpu", "cuda", "coreml", "webgpu"]
DEVICES: tuple[str, ...] = get_args(Device)
DEVICE_ENV_VAR = "HAB_DEVICE"

_CPU_EP = "CPUExecutionProvider"
_COREML_EP = "CoreMLExecutionProvider"
_CUDA_EPS = ("NvTensorRtRtxExecutionProvider", "CUDAExecutionProvider")

# CoreML compiles slowly and ran slower than the CPU on our models, so it is only used when requested explicitly
_EXCLUDED_FROM_AUTO = {"coreml"}

# PCI vendor IDs, for devices that ONNX Runtime reports without a vendor name
_VENDOR_NAMES = {0x10DE: "NVIDIA", 0x1002: "AMD", 0x1022: "AMD", 0x8086: "Intel", 0x106B: "Apple", 0x5143: "Qualcomm"}

# The [cuda] extra installs CUDA 13, which needs NVIDIA driver R580 or newer
_CUDA_INSTALL = "`pip install 'habitat-mapper[cuda]'`"

_UNAVAILABLE_HINTS = {
    "cuda": (
        "It needs an RTX 20 series or newer NVIDIA GPU, driver R580 or newer, and the CUDA libraries, which are "
        f"available on Windows and Linux. Install them with {_CUDA_INSTALL}."
    ),
    "coreml": "CoreML is only available on macOS.",
    "webgpu": "It needs a GPU with a DirectX 12 (Windows), Vulkan (Linux) or Metal (macOS) driver.",
}

# Why a device that was asked for isn't available, found by find_accelerators. Used in place of _UNAVAILABLE_HINTS
_unavailable_reasons: dict[str, str] = {}


@dataclass
class Accelerator:
    """An ONNX Runtime execution provider that can run on this machine.

    Attributes:
        device: User-facing device name, one of DEVICES other than "auto".
        ep_name: ONNX Runtime execution provider name.
        hardware: Description of the hardware the provider runs on.
        ep_devices: Devices reported by ort.get_ep_devices(). Empty for providers that are selected by name.
        options: Execution provider options.
    """

    device: str
    ep_name: str
    hardware: str
    ep_devices: tuple[Any, ...] = ()
    options: dict[str, str] = field(default_factory=dict)

    def session_kwargs(self) -> dict[str, Any]:
        """Build keyword arguments for ort.InferenceSession that run a model on this accelerator.

        Returns:
            Keyword arguments for ort.InferenceSession.
        """
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 4
        # ORT's own fallback quietly rebuilds a failed session on the CPU. create_session falls back instead, so that
        # the device that is actually used gets reported
        kwargs: dict[str, Any] = {"sess_options": sess_options, "enable_fallback": 0}
        if self.ep_devices:
            sess_options.add_provider_for_devices(list(self.ep_devices), self.options)
        elif self.ep_name == _CPU_EP:
            kwargs["providers"] = [_CPU_EP]
        else:
            kwargs["providers"] = [(self.ep_name, self.options), _CPU_EP]
        return kwargs


@contextlib.contextmanager
def _capture_native_stderr(source: str) -> Generator[None]:
    """Capture what native libraries write straight to stderr and log it at debug level.

    Dawn (used by the WebGPU plugin) and the Vulkan loader print warnings to stderr that users can't act on. Don't log
    inside this block: a loguru sink that writes to stderr would be captured too.

    Args:
        source: What is running, for the debug message.
    """
    sys.stderr.flush()
    saved_fd = os.dup(2)
    with tempfile.TemporaryFile() as captured:
        os.dup2(captured.fileno(), 2)
        try:
            yield
        finally:
            sys.stderr.flush()
            os.dup2(saved_fd, 2)
            os.close(saved_fd)
            captured.seek(0)
            output = captured.read().decode(errors="replace").strip()
            if output:
                logger.debug(f"Output from {source}:\n{output}")


class _VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = (
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.c_void_p),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.c_void_p),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.c_void_p),
    )


_VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
_VK_PHYSICAL_DEVICE_TYPE_CPU = 4


def _vulkan_device_types() -> list[int]:
    """List the VkPhysicalDeviceType of every Vulkan device.

    Returns:
        Device types, or an empty list if Vulkan isn't installed or has no devices.
    """
    try:
        vulkan = ctypes.CDLL("libvulkan.so.1")
    except OSError:
        return []
    vulkan.vkCreateInstance.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
    vulkan.vkCreateInstance.restype = ctypes.c_int32
    vulkan.vkEnumeratePhysicalDevices.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
    vulkan.vkEnumeratePhysicalDevices.restype = ctypes.c_int32
    vulkan.vkGetPhysicalDeviceProperties.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    vulkan.vkGetPhysicalDeviceProperties.restype = None
    vulkan.vkDestroyInstance.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    vulkan.vkDestroyInstance.restype = None

    instance = ctypes.c_void_p()
    create_info = _VkInstanceCreateInfo(sType=_VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO)
    if vulkan.vkCreateInstance(ctypes.byref(create_info), None, ctypes.byref(instance)) != 0:
        return []
    try:
        count = ctypes.c_uint32()
        if vulkan.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), None) != 0:
            return []
        devices = (ctypes.c_void_p * count.value)()
        if vulkan.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), devices) != 0:
            return []
        types = []
        for device in devices[: count.value]:
            # VkPhysicalDeviceProperties is 824 bytes and deviceType is its fifth uint32
            properties = (ctypes.c_uint32 * 256)()
            vulkan.vkGetPhysicalDeviceProperties(device, properties)
            types.append(properties[4])
        return types
    finally:
        vulkan.vkDestroyInstance(instance, None)


class _Luid(ctypes.Structure):
    _fields_ = (("LowPart", ctypes.c_uint32), ("HighPart", ctypes.c_int32))


class _DxgiAdapterDesc1(ctypes.Structure):
    _fields_ = (
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", ctypes.c_uint32),
        ("DeviceId", ctypes.c_uint32),
        ("SubSysId", ctypes.c_uint32),
        ("Revision", ctypes.c_uint32),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", _Luid),
        ("Flags", ctypes.c_uint32),
    )


class _Guid(ctypes.Structure):
    _fields_ = (
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_uint8 * 8),
    )


# {770aae78-f26f-4dba-a829-253c83d1b387}
_IID_IDXGIFACTORY1 = _Guid(
    0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_uint8 * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87)
)
_DXGI_ADAPTER_FLAG_SOFTWARE = 2
# The Microsoft Basic Render Driver (WARP). Windows Sandbox lists it more than once, not always with the software flag
_WARP_IDS = (0x1414, 0x8C)
# Vtable slots. IUnknown: Release=2. IDXGIFactory1: EnumAdapters1=12. IDXGIAdapter1: GetDesc1=10
_RELEASE, _ENUM_ADAPTERS1, _GET_DESC1 = 2, 12, 10
_MAX_ADAPTERS = 64


def _com_method(obj: ctypes.c_void_p, index: int, *argtypes: Any) -> Any:  # noqa: ANN401
    """Get a method of a COM object from its vtable.

    Args:
        obj: Pointer to the COM object.
        index: Slot of the method in the vtable.
        *argtypes: Argument types after the object pointer.

    Returns:
        A callable that takes the object pointer followed by the arguments and returns an HRESULT or ULONG.

    Raises:
        OSError: If not on Windows.
    """
    if sys.platform != "win32":
        raise OSError("COM is only available on Windows")
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(vtable[index])


def _release(obj: ctypes.c_void_p) -> None:
    if obj:
        _com_method(obj, _RELEASE)(obj)


def _dxgi_adapters() -> list[tuple[int, int, bool]]:
    """List the DirectX graphics adapters on Windows, including software ones such as WARP.

    WebGPU runs on Direct3D 12 on Windows. Without a vendor driver, the only adapter is the Microsoft Basic Render
    Driver (WARP), a software renderer.

    Returns:
        (vendor_id, device_id, is_software) for each adapter, or an empty list on other platforms or if DXGI can't be
        queried.
    """
    if sys.platform != "win32":
        return []
    adapters: list[tuple[int, int, bool]] = []
    factory = ctypes.c_void_p()
    try:
        dxgi = ctypes.WinDLL("dxgi")
        create_factory = dxgi.CreateDXGIFactory1
        create_factory.argtypes = (ctypes.POINTER(_Guid), ctypes.POINTER(ctypes.c_void_p))
        create_factory.restype = ctypes.c_long
        if create_factory(ctypes.byref(_IID_IDXGIFACTORY1), ctypes.byref(factory)) != 0 or not factory:
            return []
        enum_adapters = _com_method(factory, _ENUM_ADAPTERS1, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p))
        # EnumAdapters1 returns DXGI_ERROR_NOT_FOUND after the last adapter
        for index in range(_MAX_ADAPTERS):
            adapter = ctypes.c_void_p()
            if enum_adapters(factory, index, ctypes.byref(adapter)) != 0 or not adapter:
                break
            try:
                desc = _DxgiAdapterDesc1()
                get_desc = _com_method(adapter, _GET_DESC1, ctypes.POINTER(_DxgiAdapterDesc1))
                if get_desc(adapter, ctypes.byref(desc)) == 0:
                    adapters.append((desc.VendorId, desc.DeviceId, bool(desc.Flags & _DXGI_ADAPTER_FLAG_SOFTWARE)))
            finally:
                _release(adapter)
    except Exception:  # noqa: BLE001
        return []
    finally:
        try:
            _release(factory)
        except Exception:  # noqa: BLE001, S110
            pass
    return adapters


def has_hardware_gpu_for_webgpu(webgpu_devices: Sequence[Any]) -> bool:
    """Check whether WebGPU would run on a hardware GPU rather than a software renderer.

    ONNX Runtime reports the GPU it finds on the PCI bus even when the GPU's driver is missing. WebGPU then fails, or
    runs on a software renderer that is slower than the CPU: llvmpipe on Linux (Vulkan), WARP on Windows (DirectX 12).
    macOS isn't checked.

    Args:
        webgpu_devices: The WebGPU devices reported by ort.get_ep_devices().

    Returns:
        True if WebGPU should run on a hardware GPU.
    """
    if sys.platform == "linux":
        with _capture_native_stderr("the Vulkan loader"):
            types = _vulkan_device_types()
        return any(t != _VK_PHYSICAL_DEVICE_TYPE_CPU for t in types)
    if sys.platform == "win32":
        adapters = _dxgi_adapters()
        if not adapters:
            return True  # DXGI couldn't be queried, which doesn't mean the GPU is unusable
        hardware = {(vendor, device) for vendor, device, software in adapters if not software} - {_WARP_IDS}
        return any((d.device.vendor_id, d.device.device_id) in hardware for d in webgpu_devices)
    return True


def nvidia_driver_cuda_version() -> int | None:
    """Get the newest CUDA major version that the installed NVIDIA driver supports.

    Returns:
        The CUDA major version, e.g. 13, or None if there is no NVIDIA driver.
    """
    library = {"win32": "nvcuda.dll", "linux": "libcuda.so.1"}.get(sys.platform)
    if library is None:
        return None
    try:
        cuda = ctypes.CDLL(library)
    except OSError:
        return None
    version = ctypes.c_int()
    if cuda.cuDriverGetVersion(ctypes.byref(version)) != 0:
        return None
    return version.value // 1000  # e.g. 12080 for CUDA 12.8


_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
# CUDA 13 dropped Maxwell, Pascal and Volta GPUs (GTX 900 and 10 series, Quadro M and P series, Titan V)
_CUDA13_MIN_COMPUTE_CAPABILITY = (7, 5)


def nvidia_compute_capability() -> tuple[int, int] | None:
    """Get the compute capability of the first NVIDIA GPU, the one ONNX Runtime uses by default.

    Returns:
        The compute capability, e.g. (6, 1) for a Pascal GPU, or None if it can't be read.
    """
    library = {"win32": "nvcuda.dll", "linux": "libcuda.so.1"}.get(sys.platform)
    if library is None:
        return None
    try:
        cuda = ctypes.CDLL(library)
    except OSError:
        return None
    device, major, minor = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    if (
        cuda.cuInit(0) != 0
        or cuda.cuDeviceGet(ctypes.byref(device), 0) != 0
        or cuda.cuDeviceGetAttribute(ctypes.byref(major), _CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) != 0
        or cuda.cuDeviceGetAttribute(ctypes.byref(minor), _CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device) != 0
    ):
        return None
    return major.value, minor.value


def has_cudnn() -> bool:
    """Check whether cuDNN can be loaded, either from the [cuda] extra or a system install.

    The CUDA provider can create a session with only the CUDA toolkit installed, then fails on the first convolution
    when cuDNN is missing. That failure happens after the device has been chosen, so it can't fall back to another one.

    Returns:
        True if the cuDNN library can be loaded.
    """
    library = {"win32": "cudnn64_9.dll", "linux": "libcudnn.so.9"}.get(sys.platform)
    if library is None:
        return False
    try:
        ctypes.CDLL(library)
    except OSError:
        return False
    return True


def _cudnn_packages() -> set[str]:
    """List the cuDNN packages installed from PyPI, e.g. nvidia-cudnn-cu13 from the [cuda] extra.

    Returns:
        The normalized package names.
    """
    names = {(d.metadata["Name"] or "").lower().replace("_", "-") for d in distributions()}
    return {name for name in names if name.startswith("nvidia-cudnn")}


def _cuda_libraries_installed() -> bool:
    """Check whether cuDNN is installed, from the [cuda] extra or a system install, without loading CUDA.

    Returns:
        True if cuDNN is installed.
    """
    return has_cudnn() or bool(_cudnn_packages())


def _add_cudnn_to_path() -> None:
    """On Windows, put the cuDNN folder from the [cuda] extra on PATH.

    cuDNN loads its sub-libraries by name the first time it needs them, and on Windows only finds them through PATH.
    ONNX Runtime preloads a fixed list of them, which can miss ones added in newer cuDNN releases, e.g.
    cudnn_engines_tensor_ir64_9.dll in cuDNN 9.21. The model would then fail on its first convolution.
    """
    if sys.platform != "win32":
        return
    spec = find_spec("nvidia")
    if spec is None:
        return
    for root in spec.submodule_search_locations or []:
        cudnn_bin = Path(root) / "cudnn" / "bin"
        if cudnn_bin.is_dir():
            os.environ["PATH"] = f"{cudnn_bin}{os.pathsep}{os.environ.get('PATH', '')}"


def _load_cuda_libraries() -> None:
    """Load the CUDA and cuDNN libraries so ONNX Runtime can register its CUDA provider.

    ONNX Runtime prints a line for every library it fails to load. That output is discarded here; whether CUDA ended up
    usable is checked through ort.get_ep_devices() afterwards.
    """
    _add_cudnn_to_path()
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ort.preload_dlls(directory="")
        except (OSError, RuntimeError):
            pass


def _register_webgpu() -> str | None:
    """Register the WebGPU plugin execution provider if it is installed.

    Returns:
        The WebGPU execution provider name, or None if the plugin is not installed or failed to register.
    """
    if find_spec("onnxruntime_ep_webgpu") is None:
        return None

    import onnxruntime_ep_webgpu  # noqa: PLC0415

    ep_name: str = onnxruntime_ep_webgpu.get_ep_name()
    try:
        with _capture_native_stderr("the WebGPU plugin"):
            ort.register_execution_provider_library(ep_name, onnxruntime_ep_webgpu.get_library_path())
    except Exception as e:  # noqa: BLE001
        if "already registered" not in str(e).lower():
            logger.warning(f"The WebGPU plugin is installed but could not be loaded: {e}")
            return None
    return ep_name


def _preferred_webgpu_device(webgpu_devices: Sequence[Any]) -> Any:  # noqa: ANN401
    """Choose the GPU for WebGPU. The plugin rejects more than one device, e.g. both GPUs of a laptop.

    Prefers a discrete GPU, then the one Windows ranks as highest performance. ORT reports both only on Windows.

    Args:
        webgpu_devices: The WebGPU GPU devices reported by ort.get_ep_devices().

    Returns:
        The device to run on.
    """

    def rank(device: Any) -> tuple[bool, int]:  # noqa: ANN401
        metadata = getattr(device.device, "metadata", None) or {}
        performance_index = metadata.get("DxgiHighPerformanceIndex", "")
        return (
            metadata.get("Discrete") != "1",
            int(performance_index) if performance_index.isdigit() else _MAX_ADAPTERS,
        )

    return min(webgpu_devices, key=rank)


def _describe(ep_devices: Sequence[Any]) -> str:
    """Describe the hardware behind a list of ONNX Runtime EP devices.

    Returns:
        A short description, e.g. "Apple GPU, Apple NPU".
    """
    names = []
    for d in ep_devices:
        vendor = d.device.vendor or _VENDOR_NAMES.get(getattr(d.device, "vendor_id", 0), "")
        names.append(f"{vendor} {d.device.type.name}".strip())
    return ", ".join(dict.fromkeys(names))


def _explain_unused_nvidia_gpu(
    driver_cuda: int,
    ort_cuda: int,
    compute_capability: tuple[int, int] | None,
    *,
    cudnn: bool,
    other_gpu_found: bool,
) -> tuple[str, str, str]:
    """Explain why CUDA isn't used on an NVIDIA GPU, and what to do about it. CUDA is the fastest option there.

    Args:
        driver_cuda: Newest CUDA major version the NVIDIA driver supports.
        ort_cuda: CUDA major version ONNX Runtime was built for, or 0 if it was built without CUDA.
        compute_capability: The GPU's compute capability, or None if it couldn't be read.
        cudnn: Whether the CUDA libraries are installed.
        other_gpu_found: Whether another GPU accelerator, such as WebGPU, will be used instead.

    Returns:
        The log level, the message to log, and the reason to give if CUDA is requested with --device.
    """
    if compute_capability is not None and compute_capability < _CUDA13_MIN_COMPUTE_CAPABILITY:
        # Nothing to install: WebGPU or the CPU is used instead
        major, minor = compute_capability
        reason = (
            f"CUDA doesn't support your NVIDIA GPU (compute capability {major}.{minor}, older than the RTX 20 series)."
        )
        if cudnn:
            return "WARNING", f"{reason[:-1]}, so the CUDA libraries installed with `[cuda]` aren't used.", reason
        return "DEBUG", reason, reason

    if driver_cuda < max(ort_cuda, 13):
        if cudnn:
            message = (
                f"The installed CUDA libraries need CUDA {max(ort_cuda, 13)}, but your NVIDIA driver only supports "
                f"CUDA {driver_cuda}. Update the driver to R580 or newer to use CUDA."
            )
            return "WARNING", message, message
        reason = (
            f"Update the NVIDIA driver to R580 or newer and install the CUDA libraries with {_CUDA_INSTALL} to use it."
        )
        if other_gpu_found:
            message = (
                f"Update the NVIDIA driver to R580 or newer and install the CUDA libraries with {_CUDA_INSTALL} for "
                "faster processing on NVIDIA GPUs."
            )
            return "INFO", message, reason
        return "WARNING", f"An NVIDIA GPU was found, but its driver is too old for CUDA. {reason}", reason

    reason = f"Install the CUDA libraries with {_CUDA_INSTALL} to use it."
    if other_gpu_found:
        return "INFO", f"Install the CUDA libraries with {_CUDA_INSTALL} for faster processing on NVIDIA GPUs.", reason
    return "WARNING", f"An NVIDIA GPU was found but can't be used. {reason}", reason


@cache
def find_accelerators() -> tuple[Accelerator, ...]:
    """Find the execution providers that can run on this machine.

    CUDA is only considered when an NVIDIA driver new enough for the installed CUDA libraries is present, so machines
    without one never try to load CUDA.

    Returns:
        Usable accelerators in order of preference. The CPU is always last.
    """
    _unavailable_reasons.clear()
    driver_cuda = nvidia_driver_cuda_version()
    nvidia = driver_cuda is not None
    # The CUDA version ONNX Runtime was built for, e.g. "13.0". Empty for builds without CUDA
    ort_cuda = int(ort_cuda_version.split(".")[0]) if (ort_cuda_version := getattr(ort, "cuda_version", "")) else 0
    compute_capability = nvidia_compute_capability() if nvidia and ort_cuda >= 13 else None
    old_gpu = compute_capability is not None and compute_capability < _CUDA13_MIN_COMPUTE_CAPABILITY
    # The newest CUDA version both the driver and the GPU support. CUDA 13 dropped GPUs older than Turing
    supported_cuda = min(driver_cuda, 12) if driver_cuda is not None and old_gpu else driver_cuda
    driver_too_old = supported_cuda is not None and supported_cuda < ort_cuda
    if nvidia and len(cudnn_packages := _cudnn_packages()) > 1:
        # E.g. after upgrading in place from 0.16.2 or older, which installed CUDA 12. The packages share a folder,
        # and cuDNN rejects a library left behind by the other version, so CUDA fails on the first batch
        logger.warning(
            f"CUDA libraries for more than one CUDA version are installed ({', '.join(sorted(cudnn_packages))}), "
            f"which stops CUDA from working. Create a new environment and install {_CUDA_INSTALL} again."
        )
    if nvidia and not driver_too_old:
        _load_cuda_libraries()
    cudnn = nvidia and not driver_too_old and has_cudnn()
    webgpu_ep = _register_webgpu()

    ep_devices: dict[str, list[Any]] = {}
    with _capture_native_stderr("ONNX Runtime device discovery"):
        all_ep_devices = ort.get_ep_devices()
    for ep_device in all_ep_devices:
        ep_devices.setdefault(ep_device.ep_name, []).append(ep_device)
    available = ort.get_available_providers()

    found: list[Accelerator] = []
    if nvidia and not driver_too_old:
        for ep in _CUDA_EPS:
            if ep == "CUDAExecutionProvider" and not cudnn:
                continue
            if ep in ep_devices:
                found.append(Accelerator("cuda", ep, _describe(ep_devices[ep]), tuple(ep_devices[ep])))
            elif ep in available:
                found.append(Accelerator("cuda", ep, "NVIDIA GPU"))

    cuda_found = bool(found)

    webgpu_without_driver = False
    # Only hardware GPUs: ORT offers a CPU-backed WebGPU device when ORT_WEBGPU_EP_ALLOW_SOFTWARE_ADAPTER is set
    webgpu_devices = tuple(d for d in ep_devices.get(webgpu_ep or "", []) if d.device.type.name == "GPU")
    if webgpu_ep and webgpu_devices:
        if has_hardware_gpu_for_webgpu(webgpu_devices):
            webgpu_device = _preferred_webgpu_device(webgpu_devices)
            found.append(Accelerator("webgpu", webgpu_ep, _describe([webgpu_device]), (webgpu_device,)))
        else:
            webgpu_without_driver = True

    cuda_warned = False
    if driver_cuda is not None and not cuda_found:
        level, message, reason = _explain_unused_nvidia_gpu(
            driver_cuda, ort_cuda, compute_capability, cudnn=_cuda_libraries_installed(), other_gpu_found=bool(found)
        )
        logger.log(level, message)
        _unavailable_reasons["cuda"] = reason
        cuda_warned = level == "WARNING"
    if webgpu_without_driver:
        message = (
            "A GPU was found, but its graphics driver isn't installed, so WebGPU can't use it. Install your GPU "
            "vendor's driver (on Linux, including its Vulkan driver)."
        )
        _unavailable_reasons["webgpu"] = message
        # NVIDIA users were already told how to get CUDA working, their fastest option
        if not found and not cuda_warned:
            logger.warning(message)

    if _COREML_EP in ep_devices:
        found.append(
            Accelerator(
                "coreml",
                _COREML_EP,
                _describe(ep_devices[_COREML_EP]),
                tuple(ep_devices[_COREML_EP]),
                {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"},
            )
        )

    found.append(Accelerator("cpu", _CPU_EP, _describe(ep_devices.get(_CPU_EP, [])) or "CPU"))
    return tuple(found)


def describe_provider(ep_name: str) -> str:
    """Describe an execution provider for messages.

    Args:
        ep_name: ONNX Runtime execution provider name.

    Returns:
        The device and hardware, e.g. "cuda (NVIDIA GPU)", or the provider name if it isn't a known accelerator.
    """
    for accelerator in find_accelerators():
        if accelerator.ep_name == ep_name:
            return f"{accelerator.device} ({accelerator.hardware})"
    return ep_name


def select_accelerators(device: str | None = None) -> list[Accelerator]:
    """Choose which accelerators to try for a requested device.

    Args:
        device: One of DEVICES. Defaults to the HAB_DEVICE environment variable, or "auto" if it is unset.

    Returns:
        Accelerators to try in order. The CPU is always last, as a fallback.

    Raises:
        ValueError: If the device name is not recognized.
        RuntimeError: If the requested device is not available on this machine.
    """
    device = (device or os.environ.get(DEVICE_ENV_VAR) or "auto").lower()
    if device not in DEVICES:
        raise ValueError(f"Unknown device '{device}'. Choose one of: {', '.join(DEVICES)}")

    found = find_accelerators()
    cpu = found[-1]
    if device == "auto":
        return [a for a in found if a.device not in _EXCLUDED_FROM_AUTO]
    if device == "cpu":
        return [cpu]

    chosen = [a for a in found if a.device == device]
    if not chosen:
        reason = _unavailable_reasons.get(device, _UNAVAILABLE_HINTS[device])
        raise RuntimeError(f"Device '{device}' is not available on this machine. {reason}")
    return [*chosen, cpu]


def create_session(
    model_path: str | Path, device: str | None = None, *, skip: Collection[str] = ()
) -> ort.InferenceSession:
    """Create an ONNX Runtime session on the best available accelerator.

    If an accelerator fails to load the model, a warning is logged and the next one is tried, ending with the CPU.

    Args:
        model_path: Path to the ONNX model.
        device: Device to run on. See select_accelerators.
        skip: Execution providers not to try, e.g. ones that already failed to run the model.

    Returns:
        The inference session.

    Raises:
        InvalidProtobuf: If the model file is corrupt.
        NoSuchFile: If the model file does not exist.
    """
    *accelerators, cpu = select_accelerators(device)
    for accelerator in (a for a in accelerators if a.ep_name not in skip):
        try:
            with _capture_native_stderr(accelerator.ep_name):
                session = ort.InferenceSession(str(model_path), **accelerator.session_kwargs())
        except (InvalidProtobuf, NoSuchFile):
            raise
        except Exception as e:  # noqa: BLE001
            failure = str(e)
        else:
            if session.get_providers()[0] == accelerator.ep_name:
                logger.info(f"Running on {accelerator.device} ({accelerator.hardware})")
                return session
            failure = f"ONNX Runtime used {session.get_providers()} instead"

        logger.warning(
            f"Could not run the model on {accelerator.device} ({accelerator.hardware}). Trying the next device. "
            "Set DEBUG=1 to see why."
        )
        logger.debug(f"{accelerator.ep_name} failed: {failure}")

    session = ort.InferenceSession(str(model_path), **cpu.session_kwargs())
    logger.info(f"Running on cpu ({cpu.hardware})")
    return session
