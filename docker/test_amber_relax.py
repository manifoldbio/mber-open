#!/usr/bin/env python
"""
Diagnostic test script for AMBER relaxation in mber.

This script tests the AMBER/OpenMM relaxation pipeline in isolation to help
diagnose minimization failures reported in issue #11.

Tests:
1. OpenMM platform availability (CPU, CUDA)
2. Simple structure relaxation on both platforms
3. Reports detailed error information on failure

Usage:
    python docker/test_amber_relax.py

Exit codes:
    0 - All tests passed
    1 - One or more tests failed
"""

import sys
import traceback
from pathlib import Path


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def test_openmm_platforms() -> dict:
    """Test OpenMM platform availability and report capabilities."""
    print_header("OpenMM Platform Test")

    try:
        import openmm
        print(f"OpenMM version: {openmm.__version__}")
    except ImportError as e:
        print(f"ERROR: Failed to import OpenMM: {e}")
        return {"success": False, "platforms": [], "error": str(e)}

    platforms = {}
    for i in range(openmm.Platform.getNumPlatforms()):
        platform = openmm.Platform.getPlatform(i)
        name = platform.getName()
        speed = platform.getSpeed()
        platforms[name] = {
            "speed": speed,
        }
        print(f"  Platform: {name}")
        print(f"    Speed: {speed}")

    # Check for CUDA specifically
    cuda_available = "CUDA" in platforms
    print(f"\nCUDA platform available: {cuda_available}")

    if cuda_available:
        # Try to get CUDA device info
        try:
            cuda_platform = openmm.Platform.getPlatformByName("CUDA")
            print(f"CUDA platform retrieved successfully")
        except Exception as e:
            print(f"Warning: Could not get CUDA platform: {e}")

    return {"success": True, "platforms": platforms, "cuda_available": cuda_available}


def create_test_pdb() -> str:
    """Create a simple test PDB structure (alanine dipeptide-like)."""
    # Simple test structure - a small peptide
    pdb_content = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.986  -0.768  -1.210  1.00  0.00           C
ATOM      6  N   ALA A   2       3.326   1.550   0.000  1.00  0.00           N
ATOM      7  CA  ALA A   2       3.941   2.867   0.000  1.00  0.00           C
ATOM      8  C   ALA A   2       5.459   2.767   0.000  1.00  0.00           C
ATOM      9  O   ALA A   2       6.037   1.680   0.000  1.00  0.00           O
ATOM     10  CB  ALA A   2       3.499   3.723   1.184  1.00  0.00           C
ATOM     11  N   GLY A   3       6.081   3.944   0.000  1.00  0.00           N
ATOM     12  CA  GLY A   3       7.528   4.062   0.000  1.00  0.00           C
ATOM     13  C   GLY A   3       8.085   5.475   0.000  1.00  0.00           C
ATOM     14  O   GLY A   3       7.340   6.455   0.000  1.00  0.00           O
END
"""
    return pdb_content


def test_amber_relax_cpu() -> dict:
    """Test AMBER relaxation with CPU platform."""
    print_header("AMBER Relaxation Test (CPU)")

    try:
        from mber.models.colabfold.relax import relax_me
        from mber.models.alphafold.common import protein
        import tempfile

        # Create test PDB
        pdb_content = create_test_pdb()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(pdb_content)
            pdb_path = f.name

        print(f"Created test PDB at: {pdb_path}")
        print("Running AMBER relaxation with CPU...")

        # Run relaxation with CPU
        relaxed_pdb, debug_data, violations = relax_me(
            pdb_filename=pdb_path,
            use_gpu=False,
            max_iterations=100,
            tolerance=2.39,
            stiffness=10.0,
            max_outer_iterations=1
        )

        print(f"SUCCESS: CPU relaxation completed")
        print(f"  Initial energy: {debug_data['initial_energy']:.2f} kcal/mol")
        print(f"  Final energy: {debug_data['final_energy']:.2f} kcal/mol")
        print(f"  RMSD: {debug_data['rmsd']:.4f} A")
        print(f"  Attempts: {debug_data['attempts']}")

        # Cleanup
        Path(pdb_path).unlink()

        return {"success": True, "debug_data": debug_data}

    except Exception as e:
        print(f"FAILED: CPU relaxation failed with error:")
        print(f"  {type(e).__name__}: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


def test_amber_relax_gpu() -> dict:
    """Test AMBER relaxation with CUDA/GPU platform."""
    print_header("AMBER Relaxation Test (GPU/CUDA)")

    # First check if CUDA is available
    try:
        import openmm
        cuda_platform = openmm.Platform.getPlatformByName("CUDA")
        print("CUDA platform is available")
    except Exception as e:
        print(f"CUDA platform not available: {e}")
        print("Skipping GPU test")
        return {"success": True, "skipped": True, "reason": "CUDA platform not available"}

    try:
        from mber.models.colabfold.relax import relax_me
        from mber.models.alphafold.common import protein
        import tempfile

        # Create test PDB
        pdb_content = create_test_pdb()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
            f.write(pdb_content)
            pdb_path = f.name

        print(f"Created test PDB at: {pdb_path}")
        print("Running AMBER relaxation with GPU/CUDA...")

        # Run relaxation with GPU
        relaxed_pdb, debug_data, violations = relax_me(
            pdb_filename=pdb_path,
            use_gpu=True,
            max_iterations=100,
            tolerance=2.39,
            stiffness=10.0,
            max_outer_iterations=1
        )

        print(f"SUCCESS: GPU relaxation completed")
        print(f"  Initial energy: {debug_data['initial_energy']:.2f} kcal/mol")
        print(f"  Final energy: {debug_data['final_energy']:.2f} kcal/mol")
        print(f"  RMSD: {debug_data['rmsd']:.4f} A")
        print(f"  Attempts: {debug_data['attempts']}")

        # Cleanup
        Path(pdb_path).unlink()

        return {"success": True, "debug_data": debug_data}

    except Exception as e:
        print(f"FAILED: GPU relaxation failed with error:")
        print(f"  {type(e).__name__}: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


def test_with_real_structure() -> dict:
    """Test AMBER relaxation with a real predicted structure from the example."""
    print_header("AMBER Relaxation Test (Real Structure)")

    try:
        from mber.models.colabfold.relax import relax_me
        import tempfile

        # Try to find an existing structure in the examples
        example_pdb = Path("./examples/PDL1.pdb")
        if not example_pdb.exists():
            example_pdb = Path("./protocols/src/mber_protocols/stable/VHH_binder_design/examples/PDL1.pdb")

        if not example_pdb.exists():
            print("No example PDB found, skipping real structure test")
            return {"success": True, "skipped": True, "reason": "No example PDB found"}

        print(f"Testing with real structure: {example_pdb}")
        print("Running AMBER relaxation with CPU on real structure...")

        # Run relaxation
        relaxed_pdb, debug_data, violations = relax_me(
            pdb_filename=str(example_pdb),
            use_gpu=False,
            max_iterations=100,
            tolerance=2.39,
            stiffness=10.0,
            max_outer_iterations=1
        )

        print(f"SUCCESS: Real structure relaxation completed")
        print(f"  Initial energy: {debug_data['initial_energy']:.2f} kcal/mol")
        print(f"  Final energy: {debug_data['final_energy']:.2f} kcal/mol")
        print(f"  RMSD: {debug_data['rmsd']:.4f} A")
        print(f"  Attempts: {debug_data['attempts']}")

        return {"success": True, "debug_data": debug_data}

    except Exception as e:
        print(f"FAILED: Real structure relaxation failed with error:")
        print(f"  {type(e).__name__}: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}


def print_system_info() -> None:
    """Print system and environment information."""
    print_header("System Information")

    import platform
    print(f"Python version: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")

    # Check CUDA
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available (PyTorch): {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch not available")

    # Check JAX
    try:
        import jax
        print(f"JAX version: {jax.__version__}")
        print(f"JAX devices: {jax.devices()}")
    except ImportError:
        print("JAX not available")

    # Check OpenMM
    try:
        import openmm
        print(f"OpenMM version: {openmm.__version__}")
    except ImportError:
        print("OpenMM not available")


def main():
    """Run all diagnostic tests."""
    print_header("mber AMBER Relaxation Diagnostic Tests")
    print("Testing AMBER/OpenMM relaxation for issue #11 debugging")

    # Print system info
    print_system_info()

    results = {}
    all_passed = True

    # Test 1: OpenMM platforms
    results["openmm_platforms"] = test_openmm_platforms()
    if not results["openmm_platforms"]["success"]:
        all_passed = False

    # Test 2: CPU relaxation
    results["cpu_relax"] = test_amber_relax_cpu()
    if not results["cpu_relax"]["success"]:
        all_passed = False

    # Test 3: GPU relaxation
    results["gpu_relax"] = test_amber_relax_gpu()
    if not results["gpu_relax"]["success"] and not results["gpu_relax"].get("skipped"):
        all_passed = False

    # Test 4: Real structure relaxation
    results["real_structure"] = test_with_real_structure()
    if not results["real_structure"]["success"] and not results["real_structure"].get("skipped"):
        all_passed = False

    # Summary
    print_header("Test Summary")
    for test_name, result in results.items():
        status = "PASSED" if result["success"] else "FAILED"
        if result.get("skipped"):
            status = "SKIPPED"
        print(f"  {test_name}: {status}")

    if all_passed:
        print("\nAll tests passed! AMBER relaxation is working correctly.")
        return 0
    else:
        print("\nSome tests failed. See above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
