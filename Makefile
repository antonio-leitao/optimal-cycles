# Define variables
PACKAGE_NAME = recycle  # Change this to your package name
BUILD_DIR = build/
DIST_DIR = dist/
WHEEL_FILE = $(shell ls $(DIST_DIR)*.whl)

# Define commands
.PHONY: all clean build install

all: clean build install

clean:
	@echo "Cleaning build directory..."
	rm -rf $(BUILD_DIR) $(DIST_DIR)

build:
	@echo "Building the package..."
	python -m build

install: build
	@echo "Installing the package..."
	pip install --force-reinstall $(WHEEL_FILE)


