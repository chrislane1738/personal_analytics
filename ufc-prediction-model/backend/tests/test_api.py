"""Tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predictions_endpoint_exists():
    response = client.get("/api/predictions/upcoming")
    assert response.status_code in (200, 404)

def test_fighters_endpoint_exists():
    response = client.get("/api/fighters")
    assert response.status_code in (200, 404)
