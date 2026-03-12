"""
src/utils/chemistry_utils.py
-----------------------------
Shared chemistry helper functions used across the project.
Built on RDKit — the industry-standard cheminformatics library.

Usage:
    from src.utils.chemistry_utils import is_valid_smiles, get_molecular_descriptors
"""

from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, Descriptors, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import base64
from src.utils.logger import get_logger

log = get_logger(__name__)


def is_valid_smiles(smiles: str) -> bool:
    """Returns True if the SMILES string represents a valid molecule."""
    if not smiles:
        return False
    return Chem.MolFromSmiles(smiles) is not None


def canonicalize_smiles(smiles: str) -> str:
    """Standardizes a SMILES string to its canonical (unique) form."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        log.warning(f"Invalid SMILES, cannot canonicalize: {smiles}")
        return smiles
    return Chem.MolToSmiles(mol)


def get_molecular_descriptors(smiles: str) -> dict:
    """
    Computes Lipinski/Veber drug-likeness descriptors.
    Used as supplementary features in the model.

    Returns keys: mol_weight, logP, hbd, hba, tpsa, rotatable_bonds, ring_count
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}
    return {
        "mol_weight":       round(Descriptors.MolWt(mol), 3),
        "logP":             round(Descriptors.MolLogP(mol), 3),
        "hbd":              Descriptors.NumHDonors(mol),
        "hba":              Descriptors.NumHAcceptors(mol),
        "tpsa":             round(Descriptors.TPSA(mol), 3),
        "rotatable_bonds":  Descriptors.NumRotatableBonds(mol),
        "ring_count":       mol.GetRingInfo().NumRings(),
    }


def smiles_to_image_base64(smiles: str, width: int = 300, height: int = 200) -> str:
    """
    Renders a molecule as a base64 PNG image for display in Streamlit.

    Returns:
        base64 string or "" on failure
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return base64.b64encode(drawer.GetDrawingText()).decode("utf-8")
    except Exception as e:
        log.error(f"Molecule render failed: {e}")
        return ""


def compute_tanimoto_similarity(smiles_a: str, smiles_b: str) -> float:
    """
    Tanimoto (Jaccard) similarity between two molecules using Morgan fingerprints.
    1.0 = identical, 0.0 = completely different.
    Used by the recommendation engine to find similar/alternative drugs.
    """
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return 0.0
    fp_a = AllChem.GetMorganFingerprintAsBitVect(mol_a, 2, 2048)
    fp_b = AllChem.GetMorganFingerprintAsBitVect(mol_b, 2, 2048)
    return round(DataStructs.TanimotoSimilarity(fp_a, fp_b), 4)


# ── Quick test ──────────────────────────────────────────────────
if __name__ == "__main__":
    aspirin = "CC(=O)Oc1ccccc1C(=O)O"
    warfarin = "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"
    print("Valid:", is_valid_smiles(aspirin))
    print("Descriptors:", get_molecular_descriptors(aspirin))
    print("Tanimoto similarity:", compute_tanimoto_similarity(aspirin, warfarin))
