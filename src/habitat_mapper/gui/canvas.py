"""Pan and zoom view of an input image with its segmentation overlaid."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QTransform
from PySide6.QtWidgets import QGraphicsItem, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView

if TYPE_CHECKING:
    import numpy as np
    from PySide6.QtGui import QMouseEvent, QResizeEvent, QWheelEvent

_ZOOM_STEP = 1.25
_TILE_PEN = QPen(QColor(255, 255, 255, 220), 2)
_TILE_PEN.setCosmetic(True)  # Same width at every zoom level
# For the part of a tile past the edge of the image, which the model sees as padding
_OFF_IMAGE_PEN = QPen(_TILE_PEN)
_OFF_IMAGE_PEN.setStyle(Qt.PenStyle.DashLine)


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
        self._tiles: list[QGraphicsRectItem] = []
        self._fitted = True

    def set_images(self, image: np.ndarray | None, overlay: np.ndarray | None = None) -> None:
        """Show new images, zoomed to fit.

        Args:
            image: (height, width, 3) uint8 RGB array, or None to clear the canvas.
            overlay: (height, width, 4) uint8 RGBA array drawn over the image. Stretched to the image's size.
        """
        self._image.setPixmap(QPixmap() if image is None else _to_pixmap(image))
        self.set_overlay(overlay)
        self.set_tiles([])
        self.scene().setSceneRect(self._image.boundingRect())
        self.fit()

    def set_overlay(self, overlay: np.ndarray | None) -> None:
        """Replace the overlay, keeping the zoom.

        Args:
            overlay: (height, width, 4) uint8 RGBA array, stretched to the image's size, or None to remove it.
        """
        image = self._image.pixmap()
        if overlay is None or image.isNull():
            self._overlay.setPixmap(QPixmap())
            return
        self._overlay.setPixmap(_to_pixmap(overlay))
        # Previews of the input and output can differ in size by a pixel or two from rounding
        self._overlay.setTransform(
            QTransform.fromScale(image.width() / overlay.shape[1], image.height() / overlay.shape[0])
        )

    def set_tiles(self, tiles: list[tuple[int, int, int, int]]) -> None:
        """Outline the tiles being processed, dashed where they extend past the edge of the image.

        Args:
            tiles: Left, top, right and bottom of each tile in overlay pixels. Empty to remove the outlines.
        """
        for item in self._tiles:
            self.scene().removeItem(item)
        self._tiles = []
        overlay = self._overlay.pixmap()
        for left, top, right, bottom in tiles:
            inside = (left, top, min(right, overlay.width()), min(bottom, overlay.height()))
            if inside[2:] != (right, bottom):
                self._add_outline((left, top, right, bottom), _OFF_IMAGE_PEN)
            self._add_outline(inside, _TILE_PEN)

    def _add_outline(self, rect: tuple[int, int, int, int], pen: QPen) -> None:
        left, top, right, bottom = rect
        # A child of the overlay, so it is scaled the same way, but not faded with it
        item = QGraphicsRectItem(left, top, right - left, bottom - top, self._overlay)
        item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresParentOpacity)
        item.setPen(pen)
        self._tiles.append(item)

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
