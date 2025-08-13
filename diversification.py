def suggest_diversification(sector_weights, max_sector=0.4):
    suggestions = []
    for sector, weight in sector_weights.items():
        if weight > max_sector:
            suggestions.append(f"Reduce exposure in {sector} (currently {weight:.2f})")
    return suggestions
