import numpy as np

class TestEngine:
    """Simplified test version of the engine."""
    
    def calculate_entropy(self, probs: np.ndarray) -> float:
        """Calculate Shannon entropy of probability distribution."""
        eps = 1e-10
        log_probs = np.log(probs + eps)
        return -np.sum(probs * log_probs)

    def calculate_varentropy(self, probs: np.ndarray) -> float:
        """Calculate variance-entropy of probability distribution.
        
        This is a normalized probability-weighted variance:
        VarEnt(P) = (1/(n-1)) * sum(p_i * (p_i - μ)^2) where μ = mean(p)
        
        The normalization factor (n-1) accounts for the degrees of freedom
        in the probability distribution.
        """
        n = len(probs)
        mean_prob = np.mean(probs)
        squared_deviations = (probs - mean_prob) ** 2
        return np.sum(probs * squared_deviations) / (n - 1)  # Normalize by degrees of freedom

def test_entropy_calculations():
    """Test entropy and varentropy calculations with known distributions."""
    engine = TestEngine()
    
    # Test case 1: Uniform distribution
    uniform_probs = np.array([0.25, 0.25, 0.25, 0.25])
    uniform_entropy = engine.calculate_entropy(uniform_probs)
    uniform_varentropy = engine.calculate_varentropy(uniform_probs)
    print(f"\nUniform distribution tests:")
    print(f"Entropy (should be ln(4)≈1.386): {uniform_entropy:.4f}")
    print(f"Varentropy (should be 0): {uniform_varentropy:.4f}")
    assert np.isclose(uniform_entropy, np.log(4)), f"Expected entropy ln(4), got {uniform_entropy}"
    assert np.isclose(uniform_varentropy, 0, atol=1e-8), f"Expected varentropy 0, got {uniform_varentropy}"
    
    # Test case 2: Deterministic distribution
    deterministic_probs = np.array([1.0, 0.0, 0.0, 0.0])
    det_entropy = engine.calculate_entropy(deterministic_probs)
    det_varentropy = engine.calculate_varentropy(deterministic_probs)
    print(f"\nDeterministic distribution tests:")
    print(f"Entropy (should be 0): {det_entropy:.4f}")
    print(f"Varentropy (should be 0.1875): {det_varentropy:.4f}")
    assert np.isclose(det_entropy, 0, atol=1e-8), f"Expected entropy 0, got {det_entropy}"
    assert np.isclose(det_varentropy, 0.1875), f"Expected varentropy 0.1875, got {det_varentropy}"
    
    # Test case 3: Skewed distribution
    skewed_probs = np.array([0.7, 0.2, 0.05, 0.05])
    skew_entropy = engine.calculate_entropy(skewed_probs)
    skew_varentropy = engine.calculate_varentropy(skewed_probs)
    expected_entropy = -(0.7*np.log(0.7) + 0.2*np.log(0.2) + 0.05*np.log(0.05) + 0.05*np.log(0.05))
    print(f"\nSkewed distribution tests:")
    print(f"Entropy: {skew_entropy:.4f}")
    print(f"Expected entropy: {expected_entropy:.4f}")
    print(f"Varentropy: {skew_varentropy:.4f}")
    assert np.isclose(skew_entropy, expected_entropy), f"Expected entropy {expected_entropy}, got {skew_entropy}"
    
    # Test case 4: Edge case with zeros
    zero_probs = np.array([0.0, 0.0, 1.0, 0.0])
    edge_entropy = engine.calculate_entropy(zero_probs)
    edge_varentropy = engine.calculate_varentropy(zero_probs)
    print(f"\nEdge case (zeros) tests:")
    print(f"Entropy: {edge_entropy:.4f}")
    print(f"Varentropy: {edge_varentropy:.4f}")
    assert np.isfinite(edge_entropy), "Entropy should be finite for zero probabilities"
    assert np.isfinite(edge_varentropy), "Varentropy should be finite for zero probabilities"
    
    print("\nAll entropy calculation tests passed!")

if __name__ == "__main__":
    test_entropy_calculations()