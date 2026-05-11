"""
custom.py — FastAPI backend for mirroring GHL contact data to MongoDB Atlas
============================================================================
Stack   : Python 3.11+ · FastAPI · Motor (async MongoDB) · Pydantic v2
Hosting : Render (free tier compatible — includes /health ping endpoint)
Deploy  : Set env vars in Render dashboard (see .env.example section below)

Environment variables required:
    MONGODB_URI         MongoDB Atlas connection string
                        e.g. mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true
    MONGODB_DB_NAME     Database name (default: "ghl_scraper")
    CUSTOM_API_KEY      Bearer token to protect your endpoint (any secret string)
    PORT                Set automatically by Render (default: 8000 locally)

To run locally:
    pip install fastapi motor pymongo pydantic python-dotenv uvicorn
    uvicorn custom:app --reload --port 8000

To deploy on Render:
    - Build command : pip install -r requirements_custom.txt
    - Start command : uvicorn custom:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pydantic import BaseModel, Field, EmailStr, field_validator

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────

MONGODB_URI     = os.getenv("MONGODB_URI", "")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "ghl_scraper")
CUSTOM_API_KEY  = os.getenv("CUSTOM_API_KEY", "")
COLLECTION_NAME = "contacts"

log = logging.getLogger("custom_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ─── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="GHL Contact Mirror API",
    description="Receives GHL contact payloads from the scraper and stores them in MongoDB Atlas.",
    version="1.0.0",
    docs_url="/docs",      # disable in prod by setting to None
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production if needed
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ─── MongoDB lifecycle ─────────────────────────────────────────────────────────

class _DB:
    client: AsyncIOMotorClient | None = None
    db: AsyncIOMotorDatabase | None = None

_db = _DB()


@app.on_event("startup")
async def startup_db():
    if not MONGODB_URI:
        log.error("MONGODB_URI is not set — database writes will fail!")
        return
    try:
        _db.client = AsyncIOMotorClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=8000,
            connectTimeoutMS=8000,
        )
        _db.db = _db.client[MONGODB_DB_NAME]
        # Verify connectivity
        await _db.client.admin.command("ping")
        log.info(f"✓ Connected to MongoDB Atlas — db: '{MONGODB_DB_NAME}'")
        # Create index on email for fast dedup lookups
        await _db.db[COLLECTION_NAME].create_index("email", sparse=True)
        await _db.db[COLLECTION_NAME].create_index("scraped_at")
        await _db.db[COLLECTION_NAME].create_index("post_url", sparse=True)
    except Exception as e:
        log.error(f"MongoDB connection failed: {e}")


@app.on_event("shutdown")
async def shutdown_db():
    if _db.client:
        _db.client.close()
        log.info("MongoDB connection closed")


def get_db() -> AsyncIOMotorDatabase:
    if _db.db is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available",
        )
    return _db.db

# ─── Auth ──────────────────────────────────────────────────────────────────────

_bearer = HTTPBearer(auto_error=False)


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    """
    Simple Bearer-token auth. If CUSTOM_API_KEY is not set in env,
    auth is skipped (useful for initial local testing — set it in prod).
    """
    if not CUSTOM_API_KEY:
        return  # no key configured → open (dev mode)
    if credentials is None or credentials.credentials != CUSTOM_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

# ─── Pydantic models ───────────────────────────────────────────────────────────

class CustomField(BaseModel):
    key: str
    field_value: str = ""


class ContactPayload(BaseModel):
    """
    Mirrors the structure that push_to_ghl() sends to the GHL contacts API,
    plus any extra fields the scraper attaches.
    All fields are optional so partial payloads don't error out.
    """
    # Core identity
    firstName:   str = ""
    lastName:    str = ""
    name:        str = ""
    email:       str = ""
    phone:       str = ""

    # GHL metadata
    tags:        list[str] = Field(default_factory=list)
    source:      str = ""          # post URL used as source
    locationId:  str = ""          # included in create payloads

    # Custom fields array (matches GHL structure)
    customFields: list[CustomField] = Field(default_factory=list)

    # Scraper-level extras (not sent to GHL but useful to store)
    post_title:       str = ""
    post_url:         str = ""
    timestamp_text:   str = ""
    scraped_at:       str = ""
    full_body:        str = ""
    comment_author:   str = ""
    contact_email:    str = ""
    contact_phone:    str = ""

    @field_validator("email", "contact_email", mode="before")
    @classmethod
    def lowercase_email(cls, v: Any) -> str:
        return str(v).strip().lower() if v else ""

    def to_mongo_doc(self) -> dict:
        """Flatten customFields into top-level keys for easier querying."""
        doc = self.model_dump()
        # Flatten customFields → cf_<key>
        for cf in self.customFields:
            doc[f"cf_{cf.key}"] = cf.field_value
        # Store received_at in UTC
        doc["received_at"] = datetime.now(timezone.utc).isoformat()
        return doc


class ContactResponse(BaseModel):
    success:    bool
    message:    str
    contact_id: str | None = None
    action:     str = ""   # "created" | "updated" | "skipped"

# ─── Routes ───────────────────────────────────────────────────────────────────


@app.get("/health", tags=["ops"], summary="Health / keep-alive ping")
async def health():
    """
    Returns 200 OK with current UTC time.
    Use this URL as the keep-alive ping target (e.g. UptimeRobot, cron-job.org)
    to prevent Render free tier from sleeping:
        GET https://<your-service>.onrender.com/health   every 14 min
    """
    db_status = "connected" if _db.db is not None else "disconnected"
    return {
        "status":    "ok",
        "db":        db_status,
        "utc_time":  datetime.now(timezone.utc).isoformat(),
    }


@app.post(
    "/contacts",
    response_model=ContactResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["contacts"],
    summary="Create or update a contact",
    dependencies=[Depends(verify_api_key)],
)
async def create_contact(
    payload: ContactPayload,
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> ContactResponse:
    """
    Receives a contact payload (same structure as the GHL API call in main.py)
    and upserts it into MongoDB Atlas.

    Dedup logic:
      1. Match on email (if provided)
      2. Else match on phone
      3. Else match on post_url (source)
      4. Else always insert as new
    """
    coll = db[COLLECTION_NAME]
    doc  = payload.to_mongo_doc()

    # Build dedup filter — priority: email > phone > post_url
    filter_q: dict = {}
    if payload.email:
        filter_q = {"email": payload.email}
    elif payload.phone:
        filter_q = {"phone": payload.phone}
    elif payload.source:
        filter_q = {"source": payload.source}

    try:
        if filter_q:
            result = await coll.update_one(
                filter_q,
                {"$set": doc},
                upsert=True,
            )
            if result.upserted_id:
                contact_id = str(result.upserted_id)
                action     = "created"
                log.info(f"✓ Contact created | id={contact_id} | name={payload.name}")
            else:
                # Fetch _id for response
                existing = await coll.find_one(filter_q, {"_id": 1})
                contact_id = str(existing["_id"]) if existing else None
                action     = "updated"
                log.info(f"✓ Contact updated | name={payload.name}")
        else:
            # No dedup key — always insert
            result     = await coll.insert_one(doc)
            contact_id = str(result.inserted_id)
            action     = "created"
            log.info(f"✓ Contact inserted (no dedup key) | id={contact_id}")

        return ContactResponse(
            success=True,
            message=f"Contact {action} successfully",
            contact_id=contact_id,
            action=action,
        )

    except Exception as e:
        log.error(f"DB write failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}",
        )


@app.get("/contacts", tags=["contacts"], summary="List recent contacts (debug)")
async def list_contacts(
    limit: int = 20,
    db: AsyncIOMotorDatabase = Depends(get_db),
    _auth: None = Depends(verify_api_key),
):
    """Returns the most recent N contacts (for debugging/verification)."""
    coll = db[COLLECTION_NAME]
    cursor = coll.find({}, {"_id": 0}).sort("received_at", -1).limit(limit)
    contacts = await cursor.to_list(length=limit)
    return {"count": len(contacts), "contacts": contacts}


# ─── Entry point (local dev) ──────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("custom:app", host="0.0.0.0", port=port, reload=True)