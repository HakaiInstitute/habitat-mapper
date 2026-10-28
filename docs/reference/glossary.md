# Glossary

This page defines common geospatial and technical terms used throughout the Habitat-Mapper documentation.

## Geospatial Terms

**Band**
:   A layer of data in an image representing a specific wavelength of light captured by a sensor. For example, separate bands for Red, Green, Blue, and Near-Infrared (NIR) light. Multi-band images allow analysis beyond what the human eye can see.

**Coordinate Reference System (CRS)**
:   The geographic framework that defines where on Earth an image is located. Also called a "coordinate system" or "projection". Common examples include WGS84 (used by GPS) and UTM zones.

**GeoTIFF**
:   An image file format (`.tif`) that includes embedded geographic information such as coordinates, projection, and pixel size. This allows GIS software to automatically place the image in the correct location on a map.

**NIR (Near-Infrared)**
:   Light with wavelengths just beyond what humans can see (roughly 700-1400 nanometers). Many sensors capture NIR because it's useful for detecting vegetation health and water content. Healthy plants strongly reflect NIR light.

**Nodata Value**
:   A special pixel value in a raster that indicates "no information here". For example, black areas outside the survey boundary in a drone orthomosaic are often marked with a nodata value of `0` so software knows to skip them.

**Orthomosaic**
:   A geometrically corrected aerial image created by stitching together many overlapping drone or aircraft photos. Distortions from camera angles and terrain are removed, making it suitable for accurate measurements.

**Pixel Resolution**
:   The ground distance represented by one pixel in an image, often expressed in centimeters or meters. For example, "5cm resolution" means each pixel covers a 5cm × 5cm area on the ground. Smaller values mean higher detail.

**Raster**
:   An image or grid where data is stored as a matrix of pixels (cells). Each pixel has a value representing something like color, elevation, or classification. Satellite images and drone photos are rasters.

**Segmentation**
:   The process of classifying each pixel in an image into different categories (e.g., kelp, water, rock). Habitat-Mapper performs segmentation to identify where species are present.

**Tiled GeoTIFF**
:   A GeoTIFF organized internally into small square chunks (tiles) rather than as continuous rows. This structure dramatically speeds up reading small portions of large images, which is essential for Habitat-Mapper's [tiled processing](../models/how-it-works.md#processing-large-images).

## Technical Terms

**Batch Size**
:   The number of image tiles processed simultaneously by the model. Larger batch sizes can improve speed if you have sufficient GPU memory, but may cause crashes if set too high.

**CLI (Command Line Interface)**
:   A text-based way to interact with software by typing commands into a terminal, rather than clicking buttons in a graphical interface. The `hab` command is Habitat-Mapper's CLI.

**Crop Size** / **Tile Size**
:   The dimensions (in pixels) of the square windows Habitat-Mapper uses to process large images. Larger crop sizes reduce stitching artifacts but require more memory. Must be an even number (e.g., 1024, 2048, 3200).

**Inference**
:   The process of applying a trained machine learning model to new data to make predictions. When you run `hab segment`, you're performing inference on your imagery.

**Model**
:   A trained deep learning algorithm that has learned to recognize patterns in imagery. Habitat-Mapper includes multiple [models](../models/index.md) for different sensors and species (e.g., `kelp-rgb`, `mussel-gooseneck-rgb`).

**ONNX**
:   Open Neural Network Exchange, a standard format for saving machine learning models. Habitat-Mapper uses ONNX models so they can run efficiently on different hardware without needing specific deep learning frameworks.

**Virtual Environment**
:   An isolated Python installation that keeps Habitat-Mapper's dependencies separate from other software on your computer. This prevents version conflicts and makes installation cleaner.

**uint8 / uint16**
:   Data types for storing pixel values as unsigned integers. `uint8` uses values 0-255 (8 bits), while `uint16` uses 0-65535 (16 bits). Habitat-Mapper expects input images in one of these formats.

## Related Resources

- [Prepare Your Imagery](../guides/prepare-imagery.md) - Technical specifications for imagery
- [Using the Terminal](../get-started/terminal.md) - Introduction to command line basics
- [Troubleshooting & FAQ](../help/faq.md) - Answers to common questions
