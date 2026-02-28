"""Tests for FastAPI endpoints"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    import sys
    sys.path.insert(0, "/home/claude/witnessai")
    from api.main import app
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "WitnessAI"
        assert "timestamp" in data

    def test_health_version(self, client):
        resp = client.get("/api/v1/health")
        assert resp.json()["version"] == "1.0.0"


class TestIncidentsEndpoint:
    def test_list_incidents_empty(self, client):
        resp = client.get("/api/v1/incidents")
        assert resp.status_code == 200
        data = resp.json()
        assert "incidents" in data
        assert "total" in data

    def test_get_nonexistent_incident(self, client):
        resp = client.get("/api/v1/incidents/does-not-exist")
        assert resp.status_code == 404

    def test_delete_nonexistent_incident(self, client):
        resp = client.delete("/api/v1/incidents/ghost-incident")
        assert resp.status_code == 404


class TestAgentsEndpoint:
    def test_list_agents_empty(self, client):
        resp = client.get("/api/v1/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert "agents" in data
        assert "count" in data

    def test_get_nonexistent_agent(self, client):
        resp = client.get("/api/v1/agents/cam-ghost/status")
        assert resp.status_code == 404

    def test_stop_nonexistent_agent(self, client):
        resp = client.post("/api/v1/agents/cam-ghost/stop")
        assert resp.status_code == 404
