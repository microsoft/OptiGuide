from collections import defaultdict
import pandas as pd
import argparse
import gzip

def parse_mps(file_path):
    with open(file_path, 'r') as file:
        if file_path.endswith('.gz'):
            file = gzip.open(file_path, 'rt')
        lines = file.readlines()

    sections = {}
    current_section = None

    for line in lines:
        if line.startswith('*'):
            continue  # Skip comments
        if line.strip() in [
                'NAME', 'ROWS', 'COLUMNS', 'RHS', 'BOUNDS', 'ENDATA'
        ]:
            current_section = line.strip()
            sections[current_section] = []
        elif current_section:
            sections[current_section].append(line.strip())

    return sections


def extract_rows(rows):
    objective_function = {}
    constraints = {}

    for row in rows:
        parts = row.split()
        row_type = parts[0]
        row_name = parts[1]
        if row_type == 'N':
            objective_function['name'] = row_name
        else:
            constraints[row_name] = {'type': row_type, 'coefficients': {}}

    return objective_function, constraints


def extract_columns(columns, objective_function, constraints):
    objective_function['coefficients'] = {}

    for column in columns:
        parts = column.split()
        if len(parts) < 3:
            continue
        if parts[1] == "'MARKER'":
            continue  # Skip marker lines
        try:
            variable_name = parts[0]
            row_name = parts[1]
            coefficient = float(parts[2])

            if row_name == objective_function['name']:
                objective_function['coefficients'][variable_name] = coefficient
            else:
                if row_name in constraints:
                    constraints[row_name]['coefficients'][
                        variable_name] = coefficient
        except ValueError:
            continue  # Skip and continue to next line if conversion fails

    return objective_function, constraints


def extract_rhs(rhs):
    rhs_values = defaultdict(dict)

    for line in rhs:
        parts = line.split()
        constraint_name = parts[1]
        value = float(parts[2])
        rhs_values[constraint_name] = value

    return rhs_values


def extract_bounds(bounds):
    variable_bounds = defaultdict(dict)

    for line in bounds:
        parts = line.split()
        bound_type = parts[0]
        variable_name = parts[2]
        value = float(parts[3]) if len(parts) > 3 else None

        if bound_type in ['LO', 'UP']:
            variable_bounds[variable_name][bound_type] = value
        elif bound_type == 'FX':
            variable_bounds[variable_name]['LO'] = value
            variable_bounds[variable_name]['UP'] = value

    return variable_bounds


def analyze_mps(file_path):
    sections = parse_mps(file_path)

    # Extracting information from each section
    objective_function, constraints = extract_rows(sections['ROWS'])
    try:
        objective_function, constraints = extract_columns(
            sections['COLUMNS'], objective_function, constraints)
    except:
        pass
    try:
        rhs_values = extract_rhs(sections['RHS'])
    except:
        rhs_values = None
    try:
        variable_bounds = extract_bounds(sections['BOUNDS'])
    except:
        variable_bounds = None

    rst = {
        'objective_function': objective_function,
        'constraints': constraints,
    }
    if rhs_values:
        rst['rhs_values'] = rhs_values
    if variable_bounds:
        rst['variable_bounds'] = variable_bounds

    return rst


def analyze_mps_high_level(file_path):
    sections = parse_mps(file_path)

    # Extracting information from each section
    objective_function, constraints = extract_rows(sections['ROWS'])
    try:
        objective_function, constraints = extract_columns(
            sections['COLUMNS'], objective_function, constraints)
    except:
        pass
    try:
        rhs_values = extract_rhs(sections['RHS'])
    except:
        rhs_values = None
    try:
        variable_bounds = extract_bounds(sections['BOUNDS'])
    except:
        variable_bounds = None

    # Summarizing high-level information
    summary = {
        'objective_function': {
            'name': objective_function['name'],
            'num_coefficients': len(objective_function['coefficients'])
        },
        'constraints': {
            'total': len(constraints),
            'types': {
                'equality':
                sum(1 for c in constraints.values() if c['type'] == 'E'),
                'less_than_equal':
                sum(1 for c in constraints.values() if c['type'] == 'L'),
                'greater_than_equal':
                sum(1 for c in constraints.values() if c['type'] == 'G')
            }
        }
    }

    if rhs_values:
        try:
            summary['rhs_values'] = {
                'total': len(rhs_values),
                'sample':
                list(rhs_values.items())[:5]  # Displaying a sample for brevity
            }
        except:
            pass

    if variable_bounds:
        try:
            summary['variable_bounds'] = {
                'total':
                len(variable_bounds),
                'bounded':
                sum(1 for b in variable_bounds.values()
                    if 'UP' in b or 'LO' in b),
                'fixed':
                sum(1 for b in variable_bounds.values()
                    if 'UP' in b and 'LO' in b and b['UP'] == b['LO'])
            }
        except:
            pass

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, required=True)
    args = parser.parse_args()

    # Example usage:
    file_path = args.file_path
    mps_data = analyze_mps(file_path)

    df = pd.DataFrame.from_dict(mps_data, orient='index')
    df = df.transpose()
    print(df.head())
    # print a subset of df
    print(df.sample(10))
    print(analyze_mps_high_level(file_path))