"""Command-line interface for oligopoly simulations.

This module provides CLIs for running Cournot and Bertrand simulations with 
command-line arguments for market parameters, firm costs, and strategic choices.
"""

import argparse
import sys
from typing import List

from .cournot import cournot_simulation, parse_costs, parse_quantities
from .bertrand import bertrand_simulation, parse_costs as parse_bertrand_costs, parse_prices


def cournot_main() -> None:
    """CLI entry point for Cournot simulation."""
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


def bertrand_main() -> None:
    """CLI entry point for Bertrand simulation."""
    parser = argparse.ArgumentParser(
        description="Run a one-round Bertrand oligopoly simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  bertrand --alpha 120 --beta 1.2 --costs 20,20,25 --p 22,21,24
  bertrand --alpha 100 --beta 1.0 --costs 10,15 --p 12,14
        """
    )
    
    parser.add_argument(
        "--alpha", 
        type=float, 
        required=True,
        help="Intercept parameter for demand curve Q(p) = max(0, α - β*p)"
    )
    
    parser.add_argument(
        "--beta", 
        type=float, 
        required=True,
        help="Slope parameter for demand curve Q(p) = max(0, α - β*p)"
    )
    
    parser.add_argument(
        "--costs", 
        type=str, 
        required=True,
        help="Comma-separated marginal costs for each firm (e.g., '20,20,25')"
    )
    
    parser.add_argument(
        "--p", 
        type=str, 
        required=True,
        help="Comma-separated prices chosen by each firm (e.g., '22,21,24')"
    )
    
    args = parser.parse_args()
    
    try:
        # Parse inputs
        costs = parse_bertrand_costs(args.costs)
        prices = parse_prices(args.p)
        
        # Run simulation
        result = bertrand_simulation(args.alpha, args.beta, costs, prices)
        
        # Print results
        print(f"Q={result.total_demand}")
        for i, (p, q, pi) in enumerate(zip(result.prices, result.quantities, result.profits)):
            print(f"p_{i}={p}, q_{i}={q}, π_{i}={pi}")
            
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main CLI entry point - defaults to Cournot for backward compatibility."""
    cournot_main()


if __name__ == "__main__":
    main()
