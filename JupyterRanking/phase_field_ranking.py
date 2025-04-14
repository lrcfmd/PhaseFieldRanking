import numpy as np
import pandas as pd
from ElMD.ElMD import ElMD
import os

class PhaseFields:
    def __init__(self, fields):
        if os.path.exists(f'{fields}.csv'):
        #if fields in ['MMAA', 'MMXAA', 'MMMMAA']:
            print(f'Data excerpt for {fields} phase fields:') 
            self.data = pd.read_csv(f'{fields}.csv')
            print(self.data.head())
        else:
            print(f"{fields} file is not in the folder or fields are not not yet ranked.\n \
            Try selecting e.g., 'MMAA', 'MMXAA' and 'MMMMAA'")


    def select_elements(self, elements):
        """
        Select a subset of elements to focus on.
        E.g., {'Li', 'Si', 'S'}
        Args: elements (set): Subset of elements to focus on. 
        """
        allelements = self.data['phases'].apply(lambda x: set(x.split(' ')))
        issub = [x.issuperset(elements) for x in allelements]
        self.subset = self.data[issub]

    def select_elements_except(self, exclude):
        """
        Select a subset of elements to exclude
        E.g., {'Y', 'Nb', 'Pb'}
        Args: elements (set): Subset of elements to focus on. 
        """
        allelements = self.data['phases'].apply(lambda x: set(x.split(' ')))
        issub = [x.isdisjoint(exclude) for x in allelements]
        self.subset = self.data[issub]


    def distance_from(self, target):
        """
        Calculate Wasserstein distance from a target phase field.
        Args:
            target (str): Composition to calculate a distance from,
        """
        t = ElMD(target)
        self.data[f'distance_from_{target}'] = [round(t.elmd(x),2) for x in self.data['phases']]

    def assign_pareto_fronts(self, df, selected_columns):
        """
        Assign Pareto front indices to the original DataFrame.
        Args:
            df (pandas.DataFrame): The original DataFrame.
            selected_columns (list): The columns used to calculate Pareto fronts.
        Returns:
            pandas.DataFrame: A modified DataFrame with a new column indicating Pareto fronts.
        """
        name = '_'.join(selected_columns)
        print(f'Computing Pareto fronts according to {selected_columns} ...')

        data = df[selected_columns].to_numpy()  # Extract data for Pareto computation
        pareto_fronts = self.calculate_pareto_fronts_nd(data)  # Compute Pareto fronts

        # Create a new column for Pareto front indices, initialized to -1
        df[f'pareto_{name}'] = -1

        # Assign Pareto front indices
        for front_index, front in enumerate(pareto_fronts):
            for i in front:
                df.at[i, f'pareto_{name}'] = front_index

        print(df.head())
        return df

    def calculate_pareto_fronts_nd(self, data):
        """
        Calculate Pareto fronts for n-dimensional data.
        Args:
            data (numpy.ndarray): An array of shape (num_points, num_dimensions),
                                  where each row is a point in n-dimensional space.
        Returns:
            list: A list of Pareto fronts, where each front is a list of indices.
        """
        pareto_fronts = []
        remaining_points = set(range(len(data)))  # Use a set for faster lookup
     
        while remaining_points:
            current_front = set()
     
            for i in remaining_points:
                point1 = data[i]
                is_pareto = True
     
                for j in remaining_points:
                    if i == j:
                        continue
     
                    point2 = data[j]
                    # Check if point2 dominates point1
                    if all(point2 <= point1) and any(point2 < point1):
                        is_pareto = False
                        break
     
                if is_pareto:
                    current_front.add(i)
     
            pareto_fronts.append(list(current_front))
            remaining_points -= current_front  # Remove the current Pareto front
     
        return pareto_fronts

if __name__=='__main__':
    # Example usage
    fields = PhaseFields('MMXAA')

    selected_columns = ['magpie', 'mat2vec']
    df = fields.assign_pareto_fronts(fields.data, selected_columns)

    # View the updated DataFrame
    print(df.sort_values(by='pareto_front'))  # Sort by Pareto front for better visualization


    #pareto_fronts = calculate_pareto_fronts_nd(data)

    # Print Pareto fronts
    #for i, front in enumerate(pareto_fronts):
    #    print(f"Pareto Front {i + 1}: {data[front]}")

