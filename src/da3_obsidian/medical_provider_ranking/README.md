# Medical Provider Ranking Module

A Python module for ranking medical providers by similarity to OCR-extracted invoice data. Features **TF-IDF enhanced matching** that gives proper weight to rare, distinctive terms like "Rotger" to handle OCR field misassignment and improve matching accuracy.

## Key Innovation: TF-IDF Enhanced Matching

The module now uses **TF-IDF (Term Frequency-Inverse Document Frequency)** weighting to identify and prioritize rare, distinctive terms that are crucial for accurate matching. This solves the critical problem where traditional fuzzy matching treats all terms equally.

### The Rare Terms Problem

**Traditional Problem**: Common terms like "hospital", "clinic", "medical" appear frequently across providers and get the same weight as unique identifiers like "Rotger" or "Quironsalud".

**TF-IDF Solution**: Rare terms like "Rotger" receive significantly higher weights because they appear in fewer providers, making them highly distinctive for matching.

### How TF-IDF Enhancement Works

1. **Corpus Analysis**: Analyzes all provider names to identify term frequencies
2. **IDF Scoring**: Calculates Inverse Document Frequency for each term
3. **Weighted Matching**: Matches with rare terms get substantially higher scores
4. **Fallback Support**: Maintains traditional fuzzy matching as backup

## Example: Why TF-IDF Matters

**Query**: `"Clinica Rotger"` from OCR extraction  
**Target**: `"QUIRONSALUD CLINICA ROTGER"` in database

### Traditional Approach (Equal Weights)
- "Clinica" match: Standard fuzzy score
- "Rotger" match: Standard fuzzy score  
- Result: May be outranked by providers with more common term matches

### TF-IDF Enhanced Approach
- "Clinica" match: Medium weight (moderately common)
- "Rotger" match: **High weight** (very rare term)
- Result: "Rotger" match gets 2-3x higher contribution to final score

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

Required dependencies:
- `pydantic>=2.0.0` - Data validation
- `rapidfuzz>=3.0.0` - Fuzzy string matching
- `scikit-learn>=1.3.0` - TF-IDF vectorization
- `numpy>=1.24.0` - Numerical operations

### Basic Usage

```python
from provider_ranking import ProviderInfo, ProviderRanker

# OCR-extracted provider info (names may be in wrong fields)
provider_info = ProviderInfo(
    provider_legal_name="Clinica Rotger",    # OCR field assignment
    provider_localised_name="Clinica Rotger", 
    provider_network_name="quironsalud",     # Network name
    country="Spain"
)

# Provider candidates from your database
provider_candidates = [
    {
        "provider_id": "103886", 
        "name": "QUIRONSALUD CLINICA ROTGER",  # Contains rare "Rotger"
        "provider_legal_name": "INTEGRACION SANITARIA BALEAR SL",
        "country": "Spain"
    },
    {
        "provider_id": "200001",
        "name": "Quironsalud Hospital Valencia",  # Same network, different hospital
        "provider_network": "quironsalud",
        "country": "Spain"
    }
    # ... more candidates
]

# Rank providers with TF-IDF enhancement
ranker = ProviderRanker()
results = ranker.rank_providers(provider_info, provider_candidates, top_n=5)

# View results
for match in results.matches:
    print(f"{match.name}: {match.similarity_score:.3f}")
    # Expected: "QUIRONSALUD CLINICA ROTGER" ranks highest due to "Rotger" match
```

### Advanced Example

```python
# Run the complete demo
python main.py

# Test with your own data
from provider_ranking import load_provider_candidates_from_json

candidates = load_provider_candidates_from_json("my_providers.json")
results = ranker.rank_providers(provider_info, candidates)
```

## Technical Details

### TF-IDF Implementation

The enhanced similarity calculator builds a TF-IDF model from the provider corpus:

1. **Tokenization**: Extracts tokens from all provider name fields
2. **TF-IDF Vectorization**: Uses scikit-learn's `TfidfVectorizer`
3. **Token Weight Calculation**: Assigns IDF scores to individual tokens
4. **Weighted Similarity**: Combines fuzzy matching with TF-IDF weights

### Similarity Scoring

| Field | Weight | Method |
|-------|--------|---------|
| **Smart Name Match** | **55%** | **TF-IDF + Cross-field fuzzy matching** |
| Phone Match | 15% | Digit extraction + suffix matching |
| Website Match | 10% | Domain extraction + fuzzy matching |
| Zip Code Match | 10% | Exact match + partial for extended codes |
| Address Match | 5% | Combined address fuzzy matching |
| Tax/Registration | 5% | Exact matching |

### TF-IDF Configuration

```python
TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 2),    # Unigrams and bigrams
    min_df=1,              # Keep all terms (even rare ones)
    max_df=0.8,            # Remove terms in >80% of documents
    lowercase=True
)
```

## Input Data Format

### Query Provider (ProviderInfo)

```json
{
  "provider_legal_name": "Clinica Rotger",
  "provider_localised_name": "Clinica Rotger", 
  "provider_network_name": "quironsalud",
  "phone_number": "+34-971-448-500",
  "country": "Spain"
}
```

### Provider Candidates

```json
[
  {
    "provider_id": "103886",
    "name": "QUIRONSALUD CLINICA ROTGER",
    "provider_legal_name": "INTEGRACION SANITARIA BALEAR SL",
    "provider_network": "quironsalud", 
    "phone_number": "971448500.0",
    "website": "https://www.clinicarotger.com/",
    "country": "Spain"
  }
]
```

## Smart Name Matching Strategy

### Cross-Field with TF-IDF Weighting

**Query Fields**: `provider_legal_name`, `provider_localised_name`, `provider_network_name`

**Database Fields**: `name`, `provider_legal_name`, `provider_localised_name`, `provider_network`, `provider_network_name`

**Process**:
1. **Cross-Product Matching**: Every query name tested against every database name
2. **Token Extraction**: Break names into individual tokens
3. **TF-IDF Weighting**: Apply statistical weights based on term rarity
4. **Rare Term Boost**: Extra scoring boost for matching rare terms
5. **Fuzzy Fallback**: Traditional fuzzy matching when TF-IDF unavailable

**Result**: Finds correct matches even with OCR field misassignment, prioritizing rare distinctive terms.

## Performance Benefits

### TF-IDF vs Traditional Matching

**Traditional Issues**:
- Equal weight to all terms
- Common terms dominate scoring
- Rare identifiers undervalued

**TF-IDF Advantages**:
- Rare terms get 2-3x higher weights
- Distinctive identifiers prioritized
- Better handling of OCR variations
- Maintains fuzzy matching benefits

### Expected Results

For the sample query `"Clinica Rotger"` matching `"QUIRONSALUD CLINICA ROTGER"`:

- **Traditional score**: ~0.65 (good but not exceptional)
- **TF-IDF enhanced score**: ~0.85+ (excellent due to "Rotger" weight)

## Configuration Options

### Custom TF-IDF Parameters

```python
from provider_ranking import SimilarityCalculator

# Access underlying TF-IDF configuration
calculator = SimilarityCalculator()
# TF-IDF parameters set in _build_tfidf_model() method
# Modify ngram_range, min_df, max_df as needed
```

### Custom Similarity Weights

```python
from provider_ranking import SimilarityWeights

ranker = ProviderRanker()
ranker.weights = SimilarityWeights(
    name_match=0.40,           # Increase name importance
    legal_name_match=0.25,     # TF-IDF enhanced name matching
    phone_match=0.20,          # Increase phone weight
    website_match=0.10,
    zip_code_match=0.05,
    address_match=0.0,         # Disable unreliable OCR fields
    tax_registration_match=0.0
)
```

## API Reference

### Core Classes

```python
class ProviderRanker:
    def rank_providers(self, provider_info: ProviderInfo, 
                      provider_candidates: List[Dict], 
                      top_n: int = 10) -> RankingResponse

class SimilarityCalculator:
    def smart_name_similarity(self, query: ProviderInfo, 
                             candidate: Dict) -> float
    # Now includes TF-IDF weighting
```

### Response Format

```python
class RankingResponse:
    total_candidates: int
    matches: List[ProviderMatch]     # Sorted by similarity score
    query_provider: ProviderInfo

class ProviderMatch:
    provider_id: str
    name: str
    similarity_score: float          # TF-IDF enhanced score
    match_details: Dict[str, float]  # Includes name_similarity with TF-IDF
    candidate_data: Dict[str, Any]
```

## Module Architecture

```
provider_ranking/
├── models.py          # Pydantic data models
├── similarity.py      # TF-IDF enhanced similarity algorithms
├── ranker.py         # Main ranking orchestration
├── utils.py          # File and data utilities
├── __init__.py       # Package interface
├── main.py           # TF-IDF demonstration example
├── requirements.txt  # Dependencies including scikit-learn
└── README.md         # This documentation
```