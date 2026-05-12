# StaySure API Draft

## `POST /predict`

Request:
- multipart form data
- `images`: up to 5 image files
- `metadata`: JSON string with location, BHK, size, furnishing, amenities, description, cleanliness score, and bathroom flag

Response:

```json
{
  "predicted_rent": 14500,
  "confidence_score": 0.82,
  "estimated_range": {
    "min": 12800,
    "max": 16200
  },
  "similar_rooms": [],
  "explanation": []
}
```

## `POST /train`

Admin-only endpoint for triggering model retraining.

## `GET /history`

Returns authenticated user's past predictions.

