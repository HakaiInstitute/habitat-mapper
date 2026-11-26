"""Pan and zoom view of an input image with its segmentation overlaid."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap, QTransform
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QToolButton,
    QWidget,
)

from habitat_mapper.gui import theme

if TYPE_CHECKING:
    import numpy as np
    from PySide6.QtGui import QKeyEvent, QMouseEvent, QPaintEvent, QResizeEvent, QWheelEvent

_ZOOM_STEP = 1.25
_TILE_PEN = QPen(QColor(theme.OCHRE), 2)
_TILE_PEN.setCosmetic(True)  # Same width at every zoom level
# For the part of a tile past the edge of the image, which the model sees as padding
_OFF_IMAGE_PEN = QPen(_TILE_PEN)
_OFF_IMAGE_PEN.setStyle(Qt.PenStyle.DashLine)
_CONTROLS_MARGIN = 10
_OVERVIEW_SIZE = 150  # Longest side of the overview inset
_OVERVIEW_GAP = 6  # Between the overview and the zoom buttons below it
_OVERVIEW_BORDER = 1
_OVERVIEW_RADIUS = 5


def _to_pixmap(array: np.ndarray) -> QPixmap:
    height, width, channels = array.shape
    image_format = QImage.Format.Format_RGBA8888 if channels == 4 else QImage.Format.Format_RGB888
    # QImage doesn't own the array's memory, so copy it before the array goes away
    image = QImage(array.data, width, height, width * channels, image_format).copy()
    return QPixmap.fromImage(image)


class _Overview(QWidget):
    """Small copy of the whole image with the part in view outlined. Click or drag in it to move the view."""

    def __init__(self, canvas: PreviewCanvas) -> None:
        super().__init__(canvas)
        self._canvas = canvas
        self._thumbnail = QPixmap()
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Click or drag to move the view")
        self.hide()

    def set_thumbnail(self, image: QPixmap, overlay: QPixmap, opacity: float) -> None:
        """Redraw the thumbnail from the image with the overlay over it."""
        if image.isNull():
            self._thumbnail = QPixmap()
            return
        size = image.size().scaled(_OVERVIEW_SIZE, _OVERVIEW_SIZE, Qt.AspectRatioMode.KeepAspectRatio)
        self._thumbnail = QPixmap(size)
        painter = QPainter(self._thumbnail)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        target = QRectF(self._thumbnail.rect())
        painter.drawPixmap(target, image, QRectF(image.rect()))
        if not overlay.isNull():
            painter.setOpacity(opacity)
            painter.drawPixmap(target, overlay, QRectF(overlay.rect()))
        painter.end()
        border = 2 * _OVERVIEW_BORDER
        self.setFixedSize(size.width() + border, size.height() + border)
        self.update()

    def _scale(self) -> float:
        return self._thumbnail.width() / self._canvas.sceneRect().width()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        """Draw the thumbnail, dim what's out of view and outline what's in view."""
        if self._thumbnail.isNull():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        frame = QRectF(self.rect())
        # Rounded like the zoom buttons below it
        rounded = QPainterPath()
        rounded.addRoundedRect(frame, _OVERVIEW_RADIUS, _OVERVIEW_RADIUS)
        painter.setClipPath(rounded)
        painter.fillRect(frame, QColor(theme.TEAL_DEEP))
        inner = frame.adjusted(_OVERVIEW_BORDER, _OVERVIEW_BORDER, -_OVERVIEW_BORDER, -_OVERVIEW_BORDER)
        painter.drawPixmap(inner.topLeft(), self._thumbnail)

        visible = self._canvas.visible_scene_rect().intersected(self._canvas.sceneRect())
        scale = self._scale()
        view = QRectF(
            inner.left() + visible.left() * scale,
            inner.top() + visible.top() * scale,
            visible.width() * scale,
            visible.height() * scale,
        )
        outside = QPainterPath()
        outside.addRect(inner)
        inside = QPainterPath()
        inside.addRect(view)
        painter.fillPath(outside.subtracted(inside), QColor(10, 51, 53, 150))
        painter.setPen(QPen(QColor(theme.OCHRE), 1.5))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(view)
        painter.setClipping(False)
        # Same faint edge as the zoom buttons
        painter.setPen(QPen(QColor(255, 255, 255, 40), _OVERVIEW_BORDER))
        painter.drawRoundedRect(frame.adjusted(0.5, 0.5, -0.5, -0.5), _OVERVIEW_RADIUS, _OVERVIEW_RADIUS)
        painter.end()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Centre the view on the point clicked."""
        self._move_to(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Keep the view centred under the pointer while dragging."""
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._move_to(event)

    def _move_to(self, event: QMouseEvent) -> None:
        point = event.position() - QPointF(_OVERVIEW_BORDER, _OVERVIEW_BORDER)
        self._canvas.centerOn(point / self._scale())


class PreviewCanvas(QGraphicsView):
    """Shows an image with an optional overlay. Scroll to zoom, drag to pan, double click to fit, hold Space to hide."""

    def __init__(self) -> None:
        """Create an empty canvas."""
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor(theme.TEAL_DEEP))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        # Drag or use the overview to move around; scroll bars take space and crowd the zoom buttons
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Files dropped on the canvas are handled by the window
        self.setAcceptDrops(False)
        self.viewport().setAcceptDrops(False)
        self._image = QGraphicsPixmapItem()
        self._overlay = QGraphicsPixmapItem()
        self.scene().addItem(self._image)
        self.scene().addItem(self._overlay)
        self._tiles: list[QGraphicsRectItem] = []
        self._fitted = True
        self._opacity = 1.0
        self._placeholder = ("", "")
        self._drop_highlight = False
        self._controls = self._build_controls()
        self._overview = _Overview(self)
        # The hidden scroll bars still track the view, so they say when it moves or zooms
        for bar in (self.horizontalScrollBar(), self.verticalScrollBar()):
            bar.valueChanged.connect(self._update_overview)
            bar.rangeChanged.connect(self._update_overview)

    def _build_controls(self) -> QWidget:
        controls = QWidget(self)
        layout = QHBoxLayout(controls)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        for text, tip, slot in [
            ("−", "Zoom out", lambda: self._zoom(1 / _ZOOM_STEP)),
            ("+", "Zoom in", lambda: self._zoom(_ZOOM_STEP)),
            ("Fit", "Fit the whole image (or double click)", self.fit),
        ]:
            button = QToolButton()
            button.setObjectName("canvasButton")
            button.setText(text)
            button.setToolTip(tip)
            button.setAccessibleName(tip)
            button.setFocusPolicy(Qt.FocusPolicy.TabFocus)
            button.clicked.connect(slot)
            layout.addWidget(button)
        controls.adjustSize()
        controls.hide()
        return controls

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
        self._controls.setVisible(image is not None)
        self.fit()
        self._update_overview()
        self.viewport().update()

    def has_image(self) -> bool:
        """Check whether the canvas is showing an image.

        Returns:
            True if an image is shown.
        """
        return not self._image.pixmap().isNull()

    def set_placeholder(self, title: str, detail: str = "") -> None:
        """Set the message shown while there's no image.

        Args:
            title: Main line, e.g. what to do first.
            detail: Smaller second line.
        """
        self._placeholder = (title, detail)
        self.viewport().update()

    def set_drop_highlight(self, on: bool) -> None:
        """Outline the canvas while files are dragged over the window, to show they can be dropped."""
        self._drop_highlight = on
        self.viewport().update()

    def set_overlay(self, overlay: np.ndarray | None) -> None:
        """Replace the overlay, keeping the zoom.

        Args:
            overlay: (height, width, 4) uint8 RGBA array, stretched to the image's size, or None to remove it.
        """
        image = self._image.pixmap()
        if overlay is None or image.isNull():
            self._overlay.setPixmap(QPixmap())
        else:
            self._overlay.setPixmap(_to_pixmap(overlay))
            # Previews of the input and output can differ in size by a pixel or two from rounding
            self._overlay.setTransform(
                QTransform.fromScale(image.width() / overlay.shape[1], image.height() / overlay.shape[0])
            )
        self._refresh_thumbnail()

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
        self._opacity = opacity
        self._overlay.setOpacity(opacity)
        self._refresh_thumbnail()

    def fit(self) -> None:
        """Zoom to show the whole image."""
        self.fitInView(self._image, Qt.AspectRatioMode.KeepAspectRatio)
        self._fitted = True

    def _zoom(self, factor: float) -> None:
        anchor = self.transformationAnchor()
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.scale(factor, factor)
        self.setTransformationAnchor(anchor)
        self._fitted = False

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        """Zoom in or out around the mouse."""
        if not self.has_image():
            return
        factor = _ZOOM_STEP if event.angleDelta().y() > 0 else 1 / _ZOOM_STEP
        self.scale(factor, factor)
        self._fitted = False

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Zoom to fit."""
        self.fit()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        """Hide the result while Space is held, to see the image under it."""
        if event.key() == Qt.Key.Key_Space:
            if not event.isAutoRepeat():
                self._overlay.setOpacity(0)
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        """Show the result again when Space is let go."""
        if event.key() == Qt.Key.Key_Space:
            if not event.isAutoRepeat():
                self._overlay.setOpacity(self._opacity)
            return
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event: object) -> None:  # noqa: N802
        """Don't leave the result hidden if focus moves away while Space is held."""
        self._overlay.setOpacity(self._opacity)
        super().focusOutEvent(event)  # type: ignore[arg-type]

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802
        """Keep the image fitted while the window resizes, unless the user has zoomed."""
        super().resizeEvent(event)
        self._controls.move(
            self.width() - self._controls.width() - _CONTROLS_MARGIN,
            self.height() - self._controls.height() - _CONTROLS_MARGIN,
        )
        self._place_overview()
        if self._fitted:
            self.fit()

    def visible_scene_rect(self) -> QRectF:
        """Part of the image in view, in image pixels.

        Returns:
            The viewport's rectangle mapped into the scene.
        """
        return self.mapToScene(self.viewport().rect()).boundingRect()

    def _refresh_thumbnail(self) -> None:
        self._overview.set_thumbnail(self._image.pixmap(), self._overlay.pixmap(), self._opacity)
        self._place_overview()

    def _place_overview(self) -> None:
        # Right-aligned with the zoom buttons, just above them
        self._overview.move(
            self._controls.geometry().right() + 1 - self._overview.width(),
            self._controls.y() - _OVERVIEW_GAP - self._overview.height(),
        )

    def _update_overview(self) -> None:
        """Show the overview only while part of the image is out of view."""
        bars = self.horizontalScrollBar(), self.verticalScrollBar()
        zoomed_in = self.has_image() and any(bar.maximum() > bar.minimum() for bar in bars)
        self._overview.setVisible(zoomed_in)
        self._overview.update()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        """Draw the image, then the empty state message and the drop outline over it when needed."""
        super().paintEvent(event)
        if self.has_image() and not self._drop_highlight:
            return
        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        area = QRectF(self.viewport().rect()).adjusted(14, 14, -14, -14)
        if self._drop_highlight:
            pen = QPen(QColor(theme.OCHRE), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(QColor(214, 176, 74, 28))
            painter.drawRoundedRect(area, 10, 10)
        if not self.has_image():
            self._paint_placeholder(painter, area)
        painter.end()

    def _paint_placeholder(self, painter: QPainter, area: QRectF) -> None:
        title, detail = self._placeholder
        title_font = QFont(self.font())
        title_font.setPointSizeF(title_font.pointSizeF() * 1.5)
        title_font.setWeight(QFont.Weight.DemiBold)
        detail_font = QFont(self.font())
        middle = area.center().y()

        painter.setFont(title_font)
        painter.setPen(QColor(theme.WHITE))
        title_rect = QRectF(area.left(), middle - 40, area.width(), 34)
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom, title)
        if detail:
            painter.setFont(detail_font)
            painter.setPen(QColor(255, 255, 255, 170))
            detail_rect = QRectF(area.left() + 24, middle + 2, area.width() - 48, 60)
            painter.drawText(
                detail_rect,
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap,
                detail,
            )
