import numpy as np

def calculate_pareto_fronts(x, y):
    combined_data = np.column_stack((x, y))
    num_points = len(x)
    pareto_fronts = []
    
    remaining_points = list(range(num_points))
    
    while remaining_points:
        current_front = []
        for i in remaining_points:
            is_pareto = True
            x1, y1 = combined_data[i]

            for j in remaining_points:
                if i == j:
                    continue

                x2, y2 = combined_data[j]

                if x2 <= x1 and y2 <= y1 and (x2 < x1 or y2 < y1):
                    is_pareto = False
                    break

            if is_pareto:
                current_front.append(i)

        pareto_fronts.append(current_front)
        print('Pareto front:' , len(pareto_fronts))
        remaining_points = [p for p in remaining_points if p not in current_front]

    return pareto_fronts

if __name__=='__main__':
    # Example usage:
    x_values = np.array([3, 6, 4, 8, 2, 7, 5, 9, 1, 10])
    y_values = np.array([9, 7, 8, 5, 10, 6, 7, 3, 11, 2])

    pareto_fronts = calculate_pareto_fronts(x_values, y_values)

    print("Number of Pareto Fronts:", len(pareto_fronts))
    for i, front in enumerate(pareto_fronts):
       print(f"Pareto Front {i + 1}:")
       front_points = np.array([(x_values[j], y_values[j]) for j in front])
       print(front_points)

    for x,y in zip(x_values, y_values): print([x,y])

