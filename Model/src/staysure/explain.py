def explain_prediction(metadata: dict, predicted_rent: float, similar_rooms: list[dict]) -> list[str]:
    reasons = []

    location = metadata.get("location")
    if location:
        reasons.append(f"Location demand in {location} is part of the rent estimate.")

    size = metadata.get("size_sqft")
    if size:
        reasons.append(f"The listed room size of {size} sqft affects the base rent.")

    furnishing = metadata.get("furnishing_type")
    if furnishing:
        reasons.append(f"Furnishing level was considered as {furnishing}.")

    cleanliness = metadata.get("cleanliness_score")
    if cleanliness:
        reasons.append(f"Cleanliness score {cleanliness}/10 contributed to room quality.")

    if similar_rooms:
        avg_similar = sum(room["rent_price"] for room in similar_rooms) / len(similar_rooms)
        direction = "higher" if predicted_rent >= avg_similar else "lower"
        reasons.append(
            f"The prediction is {direction} than the average of the closest dataset rooms."
        )

    return reasons[:5]

