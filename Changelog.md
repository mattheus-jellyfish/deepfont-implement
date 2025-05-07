# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New `variable_aspect_ratio` augmentation for improved font recognition
- New `apply_augmentations` function for more efficient random augmentation application
- Added new dataset creation functionality:
  - `create_dataset` function to generate training data
  - `create_image` function with variable character spacing support
  - CLI commands to generate datasets from the command line

### Changed
- Improved existing augmentation functions:
  - Updated `noise_image` with better noise parameters
  - Enhanced `blur_image` with random radius between 2.5-3.5
  - Improved `affine_rotation` with proper random transformations
  - Completely replaced `gradient_fill` with actual gradient background generation
- Modified main function to use new augmentation approach
- Added better error handling for font mapping
- Restructured command-line interface with subcommands for different operations

### Fixed
- Fixed gradient augmentation that was incorrectly using Laplacian edge detection 