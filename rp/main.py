from simulation import run_simulation
from graphs import generate_all_graphs

if __name__ == "__main__":
    print("Starting Smart Transportation Simulation...")
    results = run_simulation()
    
    print("Generating Graphs...")
    generate_all_graphs(results)

    print("Done!")