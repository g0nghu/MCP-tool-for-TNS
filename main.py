from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent
import json
import numpy as np
import logging
import typing

import qtealeaves as qtl
import qtealeaves.models as qtlm
from infinite_cal import iMPS, get_local_hamiltonian_ising, get_local_hamiltonian_XXZ, pauli

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

mcp = FastMCP()

@mcp.tool()
def calculate_ground_state(model_name: str, L: int, dim: int, args: dict) -> TextContent:
    """
    Calculates the ground state energy of a 1D or 2D quantum model for a finite system.

    Args:
        model_name (str): The name of the model to simulate. 
                          Supported models: "BoseHubbard", "QuantumIsing", "XXZ"
        L (int): The length of the system (1D)
        dim (int): The dimension of the system (1 or 2)
        args: Model-specific parameters.
                - For "BoseHubbard": U (float), J (float)
                - For "QuantumIsing": J (float), g (float)
                - For "XXZ": Jx (float), Jz (float), g (float)

    Returns:
        TextContent: The ground state energy of the model and other observables in json.
    """

    my_obs = qtl.observables.TNObservables()

    if model_name == "BoseHubbard":
        if dim == 1:
            model, my_ops = qtlm.get_bose_hubbard_1d()
        elif dim == 2:
            model, my_ops = qtlm.get_bose_hubbard_2d()
        else:
            raise ValueError("BoseHubbard model only supports 1D and 2D.")
        params = {"L": L, "U": args.get("U"), "J": args.get("J")}
        my_obs += qtl.observables.TNObsLocal("<n>", "n")

    elif model_name == "QuantumIsing":
        if dim == 1:
            model, my_ops = qtlm.get_quantum_ising_1d()
        elif dim == 2:
            model, my_ops = qtlm.get_quantum_ising_2d()
        else:
            raise ValueError("QuantumIsing model only supports 1D and 2D.")
        params = {"L": L, "J": args.get("J"), "g": args.get("g")}
        my_obs += qtl.observables.TNObsLocal("<sz>", "sz")
        my_obs += qtl.observables.TNObsLocal("<sx>", "sx")

    elif model_name == "XXZ":
        if dim != 1:
            raise ValueError("XXZ model only supports 1D.")
        model, my_ops = qtlm.get_xxz_1d()
        params = {"L": L, "Jx": args.get("Jx"), "Jz": args.get("Jz"), "g": args.get("g")}
        my_obs += qtl.observables.TNObsLocal("<sz>", "sz")
        my_obs += qtl.observables.TNObsLocal("<sx>", "sx")

    else:
        raise ValueError(f"Model {model_name} is not supported.")

    my_conv = qtl.convergence_parameters.TNConvergenceParameters(
        max_iter=7, max_bond_dimension=20
    )
    
    simulation = qtl.QuantumGreenTeaSimulation(
        model,
        my_ops,
        my_conv,
        my_obs,
        tn_type=5,  # python-TTN
        has_log_file=False,
        store_checkpoints=False,
    )

    simulation.run([params], delete_existing_folder=True)
    results = simulation.get_static_obs(params)
    # return results
    # return TextContent(type="text", text=str(results["energy"]))
    results_json = _to_json(results)
    return TextContent(type="text", text=results_json)

@mcp.tool()
def calculate_uniform_ground_state(model_name: str, args: dict) -> TextContent:
    """
    Calculates the uniform ground state of a 1D infinite system.

    Args:
        model_name (str): The name of the model to simulate. 
                          Supported models: "QuantumIsing", "XXZ"
        args: Model-specific parameters.
              - For "QuantumIsing": J (float), g (float)
              - For "XXZ": Jx (float), Jz (float), g (float)

    Returns:
        TextContent: The uniform ground state of the model and other observables in json.
    """

    if model_name == "QuantumIsing":
        h = get_local_hamiltonian_ising(args["J"], args["g"])
    elif model_name == "XXZ":
        h = get_local_hamiltonian_XXZ(args["Jx"], args["Jz"], args["g"])
    else:
        raise ValueError(f"Model {model_name} is not supported.")

    imps = iMPS(Ds = 20, Dp = 2)
    imps, energy = imps.Cal_VUMPS(h, k=2, which='SR')
    results = {"energy": energy.real}
    results["sx"] = imps.Measure(pauli[1]).real
    results["sz"] = imps.Measure(pauli[3]).real
    results_json = _to_json(results)
    return TextContent(type="text", text=results_json)

def _to_json(data: dict) -> str:

    for key, value in data.items():
        if isinstance(value, np.ndarray):
            data[key] = value.tolist()
    return json.dumps(data)

def main():
    logger.info("Hello from server!")
    mcp.run("stdio")

if __name__ == "__main__":
    main()
