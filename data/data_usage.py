import yaml
from pint import UnitRegistry
from rich.console import Console
from rich.table import Table
import sys # For exiting on error
import numpy as np # Import numpy for type checking if needed

# --- Setup ---
ureg = UnitRegistry()
Q_ = ureg.Quantity
console = Console()
yaml_file_path = 'material_properties.yaml'

# --- Load Data ---
try:
    with open(yaml_file_path, 'r') as f:
        material_data = yaml.safe_load(f)
    console.print(f"[green]Successfully loaded data from:[/green] '{yaml_file_path}'")
except FileNotFoundError:
    console.print(f"[bold red]Error:[/bold red] File not found at '{yaml_file_path}'")
    sys.exit(1) # Exit the script
except yaml.YAMLError as e:
    console.print(f"[bold red]Error parsing YAML file:[/bold red]\n{e}")
    sys.exit(1) # Exit the script

# --- Function to Display Property Data ---
def display_property(material_name, property_name):
    """Loads and displays a specific material property using rich."""
    console.print(f"\n--- Accessing [bold cyan]{property_name}[/bold cyan] for [bold magenta]{material_name}[/bold magenta] ---")

    try:
        prop_data = material_data['materials'][material_name]['properties'][property_name]

        # Handle properties with temperature/value arrays
        if 'temperatures' in prop_data and 'values' in prop_data:
            temp_unit_str = prop_data.get('temperature_unit', 'dimensionless')
            val_unit_str = prop_data.get('unit', 'dimensionless')

            temps_raw = prop_data['temperatures']
            vals_raw = prop_data['values']

            # Create Pint Quantities (handle potential unit errors)
            try:
                # Ensure raw values are numeric before passing to Pint
                numeric_vals = [float(v) for v in vals_raw]
                numeric_temps = [float(t) for t in temps_raw]
                temperatures = Q_(numeric_temps, ureg(temp_unit_str))
                values = Q_(numeric_vals, ureg(val_unit_str))
            except (ValueError, TypeError) as convert_error:
                 console.print(f"[yellow]Warning: Could not convert values to numbers: {convert_error}[/yellow]")
                 console.print(f"Temperatures (raw): {temps_raw}")
                 console.print(f"Values (raw): {vals_raw}")
                 return
            except Exception as pint_error:
                 console.print(f"[yellow]Warning: Could not parse units '{temp_unit_str}' or '{val_unit_str}': {pint_error}[/yellow]")
                 console.print(f"Temperatures (raw): {temps_raw}")
                 console.print(f"Values (raw): {vals_raw}")
                 return # Stop processing this property

            table = Table(title=f"{property_name.replace('_', ' ').title()} Data")
            table.add_column(f"Temperature ({temperatures.units:~P})", style="dim", no_wrap=True) # ~P for pretty units
            table.add_column(f"Value ({values.units:~P})", justify="right")

            # Add rows, formatting Quantities using Pint's pretty format
            for temp, val in zip(temperatures, values):
                # Use ~P directly, let Pint handle numeric formatting
                table.add_row(f"{temp.magnitude:.2f}", f"{val:~P}") 

            console.print(table)

            # Example conversion (if applicable)
            if values.check('[pressure]'):
                 # Also use ~P for the converted value display
                 console.print(f"Values converted to MPa: {values.to(ureg.MPa):~P}")


        # Handle properties with list of data items (like thermal expansion)
        elif 'data' in prop_data and 'unit' in prop_data:
            val_unit_str = prop_data['unit']
            try:
                value_unit = ureg(val_unit_str)
            except Exception as pint_error:
                console.print(f"[yellow]Warning: Could not parse unit '{val_unit_str}': {pint_error}[/yellow]")
                return

            console.print(f"Unit for values: [bold]{value_unit:~P}[/bold]")
            table = Table(title=f"{property_name.replace('_', ' ').title()} Data")
            table.add_column("Temperature Range (K)", style="dim") # Assume K based on previous structure
            table.add_column(f"Value ({value_unit:~P})", justify="right")

            for item in prop_data.get('data', []):
                temp_range = item.get('temperature_range', 'N/A')
                value_raw = item.get('value')
                if value_raw is not None:
                     try:
                         # Ensure raw value is numeric
                         numeric_value = float(value_raw)
                         value_q = Q_(numeric_value, value_unit)
                         # Use ~P directly, let Pint handle numeric formatting
                         table.add_row(temp_range, f"{value_q:~P}")
                     except (ValueError, TypeError) as convert_error:
                         table.add_row(temp_range, f"[yellow]Invalid value: {value_raw}[/yellow]")
                     except Exception as pint_error: # Catch potential unit issues again
                         table.add_row(temp_range, f"[yellow]Unit error for {value_raw}[/yellow]")

                else:
                     table.add_row(temp_range, "[red]Missing[/red]")
            console.print(table)

        else:
            console.print("[yellow]Warning: Unknown data structure for this property.[/yellow]")
            console.print(prop_data)


    except KeyError:
        console.print(f"[bold red]Error:[/bold red] Could not find material '[magenta]{material_name}[/magenta]' or property '[cyan]{property_name}[/cyan]' in the data.")
    except Exception as e:
        # Print traceback for unexpected errors to help debug
        console.print_exception(show_locals=True)


# --- Main Execution ---
if __name__ == "__main__":
    # Display selected properties
    display_property('inconel_718', 'yield_strength')
    display_property('steel_316L', 'thermal_conductivity')
    display_property('al7075_t6', 'thermal_expansion_coefficient')
    display_property('inconel_718', 'specific_heat_capacity')

    # Example of trying to access non-existent data
    # display_property('titanium_ti64', 'yield_strength')
    # display_property('inconel_718', 'modulus')
