#!/bin/bash
# mber AMBER Relaxation Docker Test Script
# Tests for issue #11: AMBER relaxation minimization failure
#
# Usage:
#   ./docker/run_docker_test.sh [command]
#
# Commands:
#   build       - Build the Docker image
#   diagnostic  - Run AMBER relaxation diagnostic tests only
#   gpu         - Run 3 trajectories with GPU AMBER relaxation
#   cpu         - Run 3 trajectories with CPU AMBER relaxation
#   all         - Run all tests (diagnostic + gpu + cpu)
#   shell       - Start an interactive shell in the container
#   help        - Show this help message

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

IMAGE_NAME="mber-test:latest"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

print_header() {
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN} $1${NC}"
    echo -e "${GREEN}========================================${NC}"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi
}

check_nvidia_docker() {
    if ! docker info 2>/dev/null | grep -q "nvidia"; then
        print_warning "NVIDIA container runtime not detected"
        print_warning "GPU tests may not work correctly"
        print_warning "Install nvidia-container-toolkit for GPU support"
    fi
}

build_image() {
    print_header "Building Docker Image"
    cd "$PROJECT_DIR"
    docker build -t "$IMAGE_NAME" .
    echo -e "${GREEN}Build completed successfully${NC}"
}

run_diagnostic() {
    print_header "Running Diagnostic Tests"
    cd "$PROJECT_DIR"
    docker compose run --rm mber-diagnostic
}

run_gpu_test() {
    print_header "Running GPU Test (3 trajectories)"
    cd "$PROJECT_DIR"
    docker compose run --rm mber-gpu
}

run_cpu_test() {
    print_header "Running CPU Test (3 trajectories)"
    cd "$PROJECT_DIR"
    docker compose run --rm mber-cpu
}

run_all_tests() {
    print_header "Running All Tests"
    run_diagnostic
    run_gpu_test
    run_cpu_test
    print_header "All Tests Completed"
}

run_shell() {
    print_header "Starting Interactive Shell"
    cd "$PROJECT_DIR"
    docker run --rm -it \
        --gpus all \
        -v "$(pwd)/output:/workspace/mber-open/output" \
        "$IMAGE_NAME" \
        /bin/bash
}

show_help() {
    cat << EOF
mber AMBER Relaxation Docker Test Script
Tests for issue #11: AMBER relaxation minimization failure

Usage:
  ./docker/run_docker_test.sh [command]

Commands:
  build       - Build the Docker image
  diagnostic  - Run AMBER relaxation diagnostic tests only
  gpu         - Run 3 trajectories with GPU AMBER relaxation
  cpu         - Run 3 trajectories with CPU AMBER relaxation
  all         - Run all tests (diagnostic + gpu + cpu)
  shell       - Start an interactive shell in the container
  help        - Show this help message

Examples:
  # First time setup
  ./docker/run_docker_test.sh build

  # Quick diagnostic test
  ./docker/run_docker_test.sh diagnostic

  # Full test suite
  ./docker/run_docker_test.sh all

  # Debug interactively
  ./docker/run_docker_test.sh shell

Note: GPU tests require nvidia-container-toolkit to be installed.
EOF
}

# Main script
check_docker

case "${1:-help}" in
    build)
        check_nvidia_docker
        build_image
        ;;
    diagnostic)
        check_nvidia_docker
        run_diagnostic
        ;;
    gpu)
        check_nvidia_docker
        run_gpu_test
        ;;
    cpu)
        run_cpu_test
        ;;
    all)
        check_nvidia_docker
        run_all_tests
        ;;
    shell)
        check_nvidia_docker
        run_shell
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
