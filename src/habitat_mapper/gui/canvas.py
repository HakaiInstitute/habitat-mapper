"""Pan and zoom view of an input image with its segmentation overlaid."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QPixmap, QTransform
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

if TYPE_CHECKING:
    import numpy as np
    from PySide6.QtGui import QMouseEvent, QResizeEvent, QWheelEvent

_ZOOM_STEP = 1.25


def _to_pixmap(array: np.ndarray) -> QPixmap:
    height, width, channels = array.shape
    image_format = QImage.Format.Format_RGBA8888 if channels == 4 else QImage.Format.Format_RGB888
    # QImage doesn't own the array's memory, so copy it before the array goes away
    image = QImage(array.data, width, height, width * channels, image_format).copy()
    return QPixmap.fromImage(image)


class PreviewCanvas(QGraphicsView):
    """Shows an image with an optional overlay. Scroll to zoom, drag to pan, double click to fit."""

    def __init__(self) -> None:
        """Create an empty canvas."""
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundBrush(Qt.GlobalColor.darkGray)
        self._image = QGraphicsPixmapItem()
        self._overlay = QGraphicsPixmapItem()
        self.scene().addItem(self._image)
        self.scene().addItem(self._overlay)
        self._fitted = True

    def set_images(self, image: np.ndarray | None, overlay: np.ndarray | None = None) -> None:
        """Show new images, zoomed to fit.

        Args:
            image: (height, width, 3) uint8 RGB array, or None to clear the canvas.
            overlay: (height, width, 4) uint8 RGBA array drawn over the image. Stretched to the image's size.
        """
        self._image.setPixmap(QPixmap() if image is None else _to_pixmap(image))
        self._overlay.setPixmap(QPixmap())
        if image is not None and overlay is not None:
            self._overlay.setPixmap(_to_pixmap(overlay))
            # Previews of the input and output can differ in size by a pixel or two from rounding
            self._overlay.setTransform(
                QTransform.fromScale(image.shape[1] / overlay.shape[1], image.shape[0] / overlay.shape[0])
            )
        self.scene().setSceneRect(self._image.boundingRect())
        self.fit()

    def set_overlay_opacity(self, opacity: float) -> None:
        """Set how opaque the overlay is, from 0 to 1."""
        self._overlay.setOpacity(opacity)

    def fit(self) -> None:
        """Zoom to show the whole image."""
        self.fitInView(self._image, Qt.AspectRatioMode.KeepAspectRatio)
        self._fitted = True

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        """Zoom in or out around the mouse."""
        factor = _ZOOM_STEP if event.angleDelta().y() > 0 else 1 / _ZOOM_STEP
        self.scale(factor, factor)
        self._fitted = False

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Zoom to fit."""
        self.fit()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        """Keep the image fitted while the window resizes, unless the user has zoomed."""
        super().resizeEvent(event)
        if self._fitted:
            self.fit()
