"""Command-line interface for Cournot oligopoly simulation.

This module provides a CLI for running Cournot simulations with command-line
arguments for market parameters, firm costs, and quantities.
"""

import argparse
import sys
from typing import List

from .cournot import cournot_simulation, parse_costs, parse_quantities


def main() -> None:
    """Main CLI entry point for Cournot simulation."""
    parser = argparse.ArgumentParser(
        description="Run a one-round Cournot oligopoly simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cournot --a 100 --b 1 --costs 10,20 --q 10,20
  cournot --a 50 --b 0.5 --costs 5,10,15 --q 20,15,10
        """
    )
    
    parser.add_argument(
        "--a", 
        type=float, 
        required=True,
        help="Maximum price parameter for demand curve P = max(0, a - b*Q)"
    )
    
    parser.add_argument(
        "--b", 
        type=float, 
        required=True,
        help="Price sensitivity parameter for demand curve P = max(0, a - b*Q)"
    )
    
    parser.add_argument(
        "--costs", 
        type=str, 
        required=True,
        help="Comma-separated marginal costs for each firm (e.g., '10,20,30')"
    )
    
    parser.add_argument(
        "--q", 
        type=str, 
        required=True,
        help="Comma-separated quantities chosen by each firm (e.g., '10,20,30')"
    )
    
    args = parser.parse_args()
    
    try:
        # Parse inputs
        costs = parse_costs(args.costs)
        quantities = parse_quantities(args.q)
        
        # Run simulation
        result = cournot_simulation(args.a, args.b, costs, quantities)
        
        # Print results
        print(f"P={result.price}")
        for i, (q, pi) in enumerate(zip(result.quantities, result.profits)):
            print(f"q_{i}={q}, π_{i}={pi}")
            
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
