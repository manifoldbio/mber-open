import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
from pathlib import Path


def create_timing_summary(design_state, output_file: Optional[Path] = None):
    """
    Create a comprehensive timing summary from a design state.

    Args:
        design_state: The design state containing timing information
        output_file: Optional path to save the summary as CSV

    Returns:
        DataFrame with timing information
    """
    # Collect all timing data
    timings = []

    # Template timings
    if (
        hasattr(design_state, "template_data")
        and design_state.template_data is not None
    ):
        if hasattr(design_state.template_data, "timings"):
            for operation, time_taken in design_state.template_data.timings.items():
                timings.append(
                    {
                        "Module": "Template",
                        "Operation": operation,
                        "Time (s)": time_taken,
                        "Phase": (
                            "Main"
                            if operation in ["setup", "run", "teardown"]
                            else "Sub-step"
                        ),
                    }
                )

    # Trajectory timings
    if (
        hasattr(design_state, "trajectory_data")
        and design_state.trajectory_data is not None
    ):
        if hasattr(design_state.trajectory_data, "timings"):
            for operation, time_taken in design_state.trajectory_data.timings.items():
                timings.append(
                    {
                        "Module": "Trajectory",
                        "Operation": operation,
                        "Time (s)": time_taken,
                        "Phase": (
                            "Main"
                            if operation in ["setup", "run", "teardown"]
                            else "Sub-step"
                        ),
                    }
                )

        # Step timings (more detailed)
        if hasattr(design_state.trajectory_data, "step_timings"):
            for (
                operation,
                time_taken,
            ) in design_state.trajectory_data.step_timings.items():
                timings.append(
                    {
                        "Module": "Trajectory",
                        "Operation": operation,
                        "Time (s)": time_taken,
                        "Phase": "Detailed",
                    }
                )

    # Evaluation timings
    if (
        hasattr(design_state, "evaluation_data")
        and design_state.evaluation_data is not None
    ):
        if hasattr(design_state.evaluation_data, "timings"):
            for operation, time_taken in design_state.evaluation_data.timings.items():
                timings.append(
                    {
                        "Module": "Evaluation",
                        "Operation": operation,
                        "Time (s)": time_taken,
                        "Phase": (
                            "Main"
                            if operation in ["setup", "run", "teardown"]
                            else "Sub-step"
                        ),
                    }
                )

        # Individual binder timings
        if hasattr(design_state.evaluation_data, "binders"):
            for i, binder in enumerate(design_state.evaluation_data.binders):
                if hasattr(binder, "timings"):
                    for operation, time_taken in binder.timings.items():
                        timings.append(
                            {
                                "Module": "Evaluation",
                                "Operation": f"{operation} (Binder {i+1})",
                                "Time (s)": time_taken,
                                "Phase": "Binder",
                            }
                        )

    # Create DataFrame
    df = pd.DataFrame(timings)

    # Save to file if requested
    if output_file is not None:
        df.to_csv(output_file, index=False)

    return df


def plot_timing_summary(df: pd.DataFrame, output_file: Optional[Path] = None):
    """
    Create a visualization of timing information.

    Args:
        df: DataFrame with timing information
        output_file: Optional path to save the plot

    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Module-level summary
    module_summary = df.groupby("Module")["Time (s)"].sum().sort_values(ascending=False)
    module_summary.plot(kind="bar", ax=axes[0], color="skyblue")
    axes[0].set_title("Total Time by Module")
    axes[0].set_ylabel("Time (seconds)")

    # Plot 2: Operation-level breakdown
    # Filter to only include significant operations (>1% of total time)
    total_time = df["Time (s)"].sum()
    significant_ops = df[df["Time (s)"] > total_time * 0.01]

    # Group by module and operation
    op_summary = significant_ops.groupby(["Module", "Operation"])["Time (s)"].sum()
    op_summary = op_summary.reset_index()

    # Sort by time
    op_summary = op_summary.sort_values("Time (s)", ascending=True)

    # Create horizontal bar chart
    axes[1].barh(
        op_summary["Module"] + ": " + op_summary["Operation"],
        op_summary["Time (s)"],
        color="salmon",
    )
    axes[1].set_title("Time by Major Operation (>1% of total)")
    axes[1].set_xlabel("Time (seconds)")

    # Adjust layout
    plt.tight_layout()

    # Save if requested
    if output_file is not None:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")

    return fig


def print_timing_summary(design_state):
    """
    Print a formatted timing summary to the console.

    Args:
        design_state: The design state containing timing information
    """
    df = create_timing_summary(design_state)

    # Calculate totals
    total_time = df["Time (s)"].sum()
    module_totals = df.groupby("Module")["Time (s)"].sum()

    # Print header
    print("\n" + "=" * 80)
    print(f"{'TIMING SUMMARY':^80}")
    print("=" * 80)

    # Print module totals
    print(f"\nTotal execution time: {total_time:.2f} seconds\n")
    print(f"Module breakdown:")
    for module, time_taken in module_totals.items():
        percentage = (time_taken / total_time) * 100
        print(f"  {module:.<20} {time_taken:>8.2f}s  ({percentage:>5.1f}%)")

    # Print main phases
    print("\nMain phases:")
    main_phases = df[df["Phase"] == "Main"]
    for _, row in main_phases.iterrows():
        print(f"  {row['Module']}.{row['Operation']:.<20} {row['Time (s)']:>8.2f}s")

    # Print top 10 most time-consuming operations
    print("\nTop 10 most time-consuming operations:")
    top_10 = df.sort_values("Time (s)", ascending=False).head(10)
    for i, (_, row) in enumerate(top_10.iterrows(), 1):
        percentage = (row["Time (s)"] / total_time) * 100
        print(
            f"  {i:2}. {row['Module']}.{row['Operation']:.<25} {row['Time (s)']:>8.2f}s  ({percentage:>5.1f}%)"
        )

    print("=" * 80)


def add_timing_to_notebook(design_state, display_function=None):
    """
    Create an interactive timing summary for a Jupyter notebook.

    Args:
        design_state: The design state containing timing information
        display_function: Function to display the output (usually IPython.display.display)

    Returns:
        DataFrame with timing information
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display, HTML
    except ImportError:
        print("ipywidgets and IPython are required for interactive display")
        return create_timing_summary(design_state)

    if display_function is None:
        display_function = display

    # Create summary
    df = create_timing_summary(design_state)

    # Calculate totals
    total_time = df["Time (s)"].sum()

    # Create title
    title_html = HTML(f"<h2>Timing Summary (Total: {total_time:.2f}s)</h2>")
    display_function(title_html)

    # Create tabs for different views
    tab = widgets.Tab()

    # Tab 1: Summary by module
    module_summary = df.groupby("Module")["Time (s)"].sum().reset_index()
    module_summary["Percentage"] = module_summary["Time (s)"] / total_time * 100
    module_summary = module_summary.sort_values("Time (s)", ascending=False)

    # Tab 2: Main phases
    main_phases = df[df["Phase"] == "Main"].copy()
    main_phases["Percentage"] = main_phases["Time (s)"] / total_time * 100

    # Tab 3: All operations
    all_ops = df.copy()
    all_ops["Percentage"] = all_ops["Time (s)"] / total_time * 100
    all_ops = all_ops.sort_values("Time (s)", ascending=False)

    # Create output widgets
    tab.children = [widgets.Output(), widgets.Output(), widgets.Output()]

    tab.set_title(0, "By Module")
    tab.set_title(1, "Main Phases")
    tab.set_title(2, "All Operations")

    display_function(tab)

    with tab.children[0]:
        display_function(module_summary)

    with tab.children[1]:
        display_function(main_phases)

    with tab.children[2]:
        display_function(all_ops)

    return df
