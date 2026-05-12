# StaySure AI Project Plan

## Product Goal

StaySure AI is a prop-tech rent prediction platform that estimates monthly rent for rooms, PGs, flats, and hostels using room images plus structured metadata.

The product will be built in phases so the machine learning system is validated before the full web application is connected.

## Phase 1: Model Training Pipeline

Status: In progress

Deliverables:
- Kaggle-ready training script.
- Google Sheets or local CSV dataset loading.
- Up to 5 room image inputs per listing.
- Metadata preprocessing for location, BHK, furnishing, size, cleanliness, bathroom, and amenities.
- CNN image branch.
- Metadata neural network branch.
- Feature concatenation and rent regression head.
- K-fold cross validation.
- MAE, RMSE, R2, and MAPE metrics.
- Saved `.keras` model.
- Saved metadata preprocessor.
- Similar room embedding index.
- CLI inference script for testing trained artifacts.

## Phase 2: Backend API

Deliverables:
- FastAPI backend.
- JWT signup/login.
- PostgreSQL database models.
- `/predict`, `/train`, `/similar-rooms`, `/history`, `/login`, `/signup` endpoints.
- Upload validation for up to 5 images.
- Model inference service.
- Prediction history storage.
- Admin dataset management endpoints.
- OpenAPI documentation.
- Docker support.

## Phase 3: Frontend Application

Deliverables:
- React + Tailwind CSS app.
- Landing page.
- Login/signup pages.
- Drag-and-drop room upload page.
- Metadata form.
- Prediction result page.
- Dashboard with history and analytics.
- Admin panel.
- Axios API integration.
- Framer Motion transitions.
- Responsive UI.

## Phase 4: Similar Room Matching

Deliverables:
- Embedding generation from trained model.
- Cosine similarity search.
- Weighted matching using visual similarity, metadata similarity, and location similarity.
- Top similar rooms with similarity percentage and rent comparison.
- API integration with prediction response.

## Phase 5: AI Explanation and Quality Scoring

Deliverables:
- Explanation text for predicted rent.
- Room quality score.
- Cleanliness score support.
- Confidence score.
- Estimated rent range.
- Future-ready interface for Gemini Vision and EfficientNet.

## Phase 6: Deployment

Deliverables:
- Frontend deployment config for Vercel.
- Backend deployment config for Render, Railway, or AWS.
- Docker and Docker Compose setup.
- Environment variable documentation.
- Production setup guide.

## Current Build Order

1. Build and verify the model training pipeline.
2. Train the first model in Kaggle using the dataset CSV and room images.
3. Review metrics and generated artifacts.
4. Connect the trained model to the FastAPI backend.
5. Build the React frontend around the working API.

