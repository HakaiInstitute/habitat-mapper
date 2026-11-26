"""Configuration models for segmentation processing and ONNX models."""

from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from loguru import logger
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ImportString,
    PositiveInt,
)

from habitat_mapper.accelerators import Device  # noqa: TC001 (Pydantic needs it at runtime)
from habitat_mapper.utils import download_dependencies, get_local_model_dir

if TYPE_CHECKING:
    from habitat_mapper.reader import ImageReader


def _is_even(value: int) -> int:
    if value % 2:
        raise ValueError("Tile size must be an even number.")
    return value


def _is_odd_or_zero(value: int) -> int:
    if value and value % 2 == 0:
        raise ValueError("Kernel sizes must be an odd number, or 0 to turn the filter off.")
    return value


def _all_positive(value: list[int]) -> list[int]:
    if any(v < 1 for v in value):
        raise ValueError("Band numbers start at 1.")
    return value


# The rules for each processing option, shared by SegmentOptions, the command line and the desktop app
TileSize = Annotated[int, Field(gt=0), AfterValidator(_is_even)]
KernelSize = Annotated[int, Field(ge=0), AfterValidator(_is_odd_or_zero)]
BandOrder = Annotated[list[int], Field(min_length=1), AfterValidator(_all_positive)]


class OutputClass(BaseModel):
    """A class value a model writes to its output, with a name and colour for showing it."""

    value: Annotated[int, Field(gt=0, description="Pixel value of the class in the output raster")]
    name: Annotated[str, "Plain name of the class, e.g. 'Giant kelp'"]
    colour: Annotated[str, Field(pattern=r"^#[0-9A-Fa-f]{6}$", description="Colour to draw the class in, as #RRGGBB")]


class ModelConfig(BaseModel):
    """ONNXModel configuration for ONNX semantic segmentation models.

    Configs are frozen, since one config is shared by every model made from it. Use `model_copy(update={...})` to make
    a changed copy.
    """

    model_config = ConfigDict(frozen=True, validate_default=True)

    model_cls: Annotated[ImportString, "For dynamic loading of model class in registry"] = (
        "habitat_mapper.model.ONNXModel"
    )
    name: Annotated[str, "The name of the model for the model registry"]
    description: Annotated[str | None, "Brief description of the model for the model registry"] = None
    revision: Annotated[
        str,
        Field(
            pattern=r"^\d{8}(-\S+)?$",
            description="Model revision in YYYYMMDD or YYYYMMDD-variant format",
        ),
    ]
    beta: bool = False
    classes: Annotated[
        list[OutputClass],
        "Classes the model outputs, other than background (0). Used to colour and label previews.",
    ] = Field(default_factory=list)
    dependencies: Annotated[
        list[str],
        "List of files to download (URLs or local paths). Downloaded to model-specific cache subdirectory.",
    ]
    model_filename: Annotated[
        str,
        "Name of the file in dependencies list that is the ONNX model (e.g., 'model.onnx')",
    ]
    input_channels: PositiveInt = 3
    activation: (
        Annotated[
            Literal["sigmoid", "softmax"],
            "Final activation to apply to model outputs. "
            "If None, it is assumed that the output of the ONNX model are probabilities",
        ]
        | None
    ) = None

    normalization: (
        Literal[
            "standard",
            "image",
            "image_per_channel",
            "min_max",
            "min_max_per_channel",
        ]
        | None
    ) = "standard"
    mean: tuple[float, ...] | None = (0.485, 0.456, 0.406)
    std: tuple[float, ...] | None = (0.229, 0.224, 0.225)

    max_pixel_value: Annotated[
        float | Literal["auto"],
        "Value to scale input pixels by prior to normalization, or 'auto' to infer from the image datatype.",
    ] = "auto"
    default_output_value: Annotated[
        int,
        (
            "Value the model should output for tiles that are all the same colour (used to speed up processing by "
            "skipping background areas)"
        ),
    ] = 0
    nodata_value: Annotated[int, "The nodata value for the output raster"] = 0

    reader_cls: Annotated[
        ImportString,
        "Fully qualified class name of the ImageReader to use (e.g., 'habitat_mapper.reader.TIFFReader')",
    ] = "habitat_mapper.reader.TIFFReader"
    reader_kwargs: Annotated[
        dict[str, Any],
        "Additional keyword arguments to pass to the reader constructor",
    ] = Field(default_factory=dict)

    def max_pixel_value_for(self, dtype: str) -> float:
        """The value to scale an image's pixels by, resolving "auto" from the image's data type.

        Args:
            dtype: Data type of the image, e.g. "uint8".

        Returns:
            The configured max_pixel_value, or the largest value of dtype when it is "auto".

        Raises:
            ValueError: If max_pixel_value is "auto" and the data type isn't supported.
        """
        if self.max_pixel_value != "auto":
            return self.max_pixel_value
        if dtype == "uint8":
            return 255
        if dtype == "uint16":
            return 65535
        if dtype in ("float16", "float32", "float64"):
            logger.warning(
                "Input image has float data type. Ensure pixel values are in range [0,1] to avoid unexpected results."
            )
            return 1.0
        raise ValueError(
            f"Unsupported image data type {dtype} for model. Convert your image to uint8 or uint16 before processing."
        )

    def get_local_dependency_paths(self, quiet: bool = False) -> dict[str, Path]:
        """Get local paths to all dependency files.

        Downloads dependencies in parallel if they are URLs and not yet cached.

        Args:
            quiet: If True, suppress download progress bars.

        Returns:
            Dict mapping filenames to local paths for all dependencies
        """
        dep_paths = download_dependencies(self, quiet=quiet)
        # Create a dict mapping filename to path for easy lookup
        return {path.name: path for path in dep_paths}

    def get_local_model_path(self, quiet: bool = False) -> Path:
        """Get the local path to the ONNX model file.

        Downloads all dependencies (including the model) if needed.

        Args:
            quiet: If True, suppress download progress bars.

        Returns:
            Path to the local ONNX model file

        Raises:
            ValueError: If model_filename not found in dependencies
        """
        # Download all dependencies (this will be cached on subsequent calls)
        dep_paths = self.get_local_dependency_paths(quiet=quiet)

        # Find the model file by name
        if self.model_filename not in dep_paths:
            raise ValueError(
                f"Model file '{self.model_filename}' not found in dependencies. "
                f"Available files: {list(dep_paths.keys())}"
            )

        model_path = dep_paths[self.model_filename]

        # Validate it's an ONNX file
        if not model_path.suffix.lower() == ".onnx":
            logger.warning(f"Model file does not have .onnx extension: {model_path}")

        return model_path

    def get_reader(self, input_path: str | Path) -> "ImageReader":
        """Instantiate the configured ImageReader for the given input.

        Args:
            input_path: Path to the input file or directory

        Returns:
            An instantiated ImageReader configured for this model

        Raises:
            TypeError: If input_path is not compatible with the reader
        """
        # Handle special __model_cache__ string
        model_cache_dir = get_local_model_dir(self.name, self.revision)
        kwargs = {
            k: (v.replace("__model_cache__", str(model_cache_dir)) if isinstance(v, str) else v)
            for k, v in self.reader_kwargs.items()
        }
        try:
            return self.reader_cls(input_path, **kwargs)
        except TypeError as e:
            # Re-raise with additional context about the reader and provided arguments
            raise TypeError(f"Cannot instantiate {self.reader_cls.__name__} with the provided arguments: {e}") from e


class SegmentOptions(BaseModel):
    """Options for segmenting an image, the same as the options of `hab segment`.

    Every option is checked here, so invalid values raise a `pydantic.ValidationError` (a `ValueError`).
    """

    model_config = ConfigDict(frozen=True)

    crop_size: Annotated[TileSize | None, "Tile size, or None for the model's preferred size (or 1024)"] = None
    batch_size: PositiveInt = 1
    blur_kernel_size: Annotated[KernelSize, "Median blur kernel size, or 0 for none"] = 0
    morph_kernel_size: Annotated[KernelSize, "Morphological open and close kernel size, or 0 for none"] = 0
    band_order: Annotated[BandOrder | None, "Bands to read, starting at 1, or None for the first bands in order"] = None
    device: Annotated[Device | None, "Hardware to run on, or None for $HAB_DEVICE or auto"] = None


class ProcessingConfig(BaseModel):
    """SegmentOptions resolved against a model: the tile size and bands are known."""

    model_config = ConfigDict(frozen=True)

    crop_size: TileSize
    band_order: BandOrder
    batch_size: PositiveInt = 1
    blur_kernel_size: KernelSize = 0
    morph_kernel_size: KernelSize = 0

    @property
    def stride(self) -> int:
        """Distance between tiles, which overlap by half."""
        return self.crop_size // 2

    @property
    def apply_morphological_ops(self) -> bool:
        """Whether to apply morphological operations."""
        return self.morph_kernel_size > 1

    @property
    def apply_median_blur(self) -> bool:
        """Whether to apply median blur."""
        return self.blur_kernel_size > 1


if __name__ == "__main__":
    config = ModelConfig(
        name="kelp_ps8b",
        revision="20250626",
        description="Kelp segmentation model for 8-band PlanetScope imagery.",
        dependencies=[
            "https://hakai-triton-models-bf426c24.s3.us-east-1.amazonaws.com/kelp_segmentation_ps8b_model/1/model.onnx"
        ],
        model_filename="model.onnx",
        normalization="standard",
        activation=None,
        mean=(1720.0, 1715.0, 1913.0, 2088.0, 2274.0, 2290.0, 2613.0, 3970.0),
        std=(747.0, 698.0, 739.0, 768.0, 849.0, 868.0, 849.0, 914.0),
        max_pixel_value=1.0,
    )

    with open(Path("./default_configs") / f"{config.name}.json", "w") as f:
        f.write(config.model_dump_json(indent=4, exclude_none=True))
