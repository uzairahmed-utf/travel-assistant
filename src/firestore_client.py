from __future__ import annotations

import random
import string
from datetime import datetime, timezone

from google.cloud.firestore_v1 import AsyncClient

from models import (
    Booking,
    BookingStatus,
    CabinClass,
    CustomerProfile,
    FareBreakdown,
    FlightSegment,
)

_db: AsyncClient | None = None


def _get_db() -> AsyncClient:
    global _db
    if _db is None:
        _db = AsyncClient(project="receiptflow-00")
    return _db


def _generate_pnr() -> str:
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


def _booking_to_dict(booking: Booking) -> dict:
    return {
        "pnr": booking.pnr,
        "customer_id": booking.customer_id,
        "flight": {
            "flight_number": booking.flight.flight_number,
            "airline": booking.flight.airline,
            "origin": booking.flight.origin,
            "destination": booking.flight.destination,
            "departure_time": booking.flight.departure_time,
            "arrival_time": booking.flight.arrival_time,
            "duration_minutes": booking.flight.duration_minutes,
            "aircraft": booking.flight.aircraft,
        },
        "fare": {
            "base_fare": booking.fare.base_fare,
            "taxes": booking.fare.taxes,
            "fuel_surcharge": booking.fare.fuel_surcharge,
            "total": booking.fare.total,
            "cabin_class": booking.fare.cabin_class.value,
        },
        "status": booking.status.value,
        "created_at": booking.created_at,
    }


def _dict_to_booking(data: dict) -> Booking:
    flight = data["flight"]
    fare = data["fare"]
    return Booking(
        pnr=data["pnr"],
        customer_id=data.get("customer_id", ""),
        flight=FlightSegment(
            flight_number=flight["flight_number"],
            airline=flight["airline"],
            origin=flight["origin"],
            destination=flight["destination"],
            departure_time=flight["departure_time"],
            arrival_time=flight["arrival_time"],
            duration_minutes=flight["duration_minutes"],
            aircraft=flight["aircraft"],
        ),
        fare=FareBreakdown(
            base_fare=fare["base_fare"],
            taxes=fare["taxes"],
            fuel_surcharge=fare["fuel_surcharge"],
            total=fare["total"],
            cabin_class=CabinClass(fare["cabin_class"]),
        ),
        status=BookingStatus(data["status"]),
        created_at=data["created_at"],
    )


async def save_booking(booking: Booking) -> None:
    db = _get_db()
    # Generate unique PNR
    while True:
        pnr = _generate_pnr()
        doc = await db.collection("bookings").document(pnr).get()
        if not doc.exists:
            break
    booking.pnr = pnr
    booking.created_at = datetime.now(timezone.utc).isoformat()
    await db.collection("bookings").document(pnr).set(_booking_to_dict(booking))


async def get_booking(pnr: str) -> Booking | None:
    db = _get_db()
    doc = await db.collection("bookings").document(pnr).get()
    if not doc.exists:
        return None
    return _dict_to_booking(doc.to_dict())


async def update_booking_status(pnr: str, status: BookingStatus) -> Booking | None:
    db = _get_db()
    doc_ref = db.collection("bookings").document(pnr)
    doc = await doc_ref.get()
    if not doc.exists:
        return None
    await doc_ref.update({"status": status.value})
    data = doc.to_dict()
    data["status"] = status.value
    return _dict_to_booking(data)


async def authenticate_customer(name: str, pin: str) -> CustomerProfile | None:
    db = _get_db()
    query = db.collection("customers").where("name", "==", name).where("pin", "==", pin)
    docs = []
    async for doc in query.stream():
        docs.append(doc)
    if not docs:
        return None
    data = docs[0].to_dict()
    return CustomerProfile(
        customer_id=docs[0].id,
        name=data["name"],
        pin=data["pin"],
        email=data["email"],
        phone=data["phone"],
        date_of_birth=data.get("date_of_birth", ""),
        gender=data.get("gender", ""),
        passport_number=data.get("passport_number", ""),
        bookings=data.get("bookings", []),
    )


async def create_customer(
    name: str,
    email: str,
    phone: str,
    pin: str,
    date_of_birth: str = "",
    gender: str = "",
    passport_number: str = "",
    pnr: str | None = None,
) -> CustomerProfile:
    db = _get_db()
    bookings = [pnr] if pnr else []
    data = {
        "name": name,
        "pin": pin,
        "email": email,
        "phone": phone,
        "date_of_birth": date_of_birth,
        "gender": gender,
        "passport_number": passport_number,
        "bookings": bookings,
    }
    _, doc_ref = await db.collection("customers").add(data)
    return CustomerProfile(
        customer_id=doc_ref.id,
        name=name,
        pin=pin,
        email=email,
        phone=phone,
        date_of_birth=date_of_birth,
        gender=gender,
        passport_number=passport_number,
        bookings=bookings,
    )


async def update_customer(customer_id: str, **fields) -> None:
    db = _get_db()
    await db.collection("customers").document(customer_id).update(fields)


async def add_booking_to_customer(customer_id: str, pnr: str) -> None:
    from google.cloud.firestore_v1 import ArrayUnion

    db = _get_db()
    await db.collection("customers").document(customer_id).update(
        {"bookings": ArrayUnion([pnr])}
    )


async def get_customer_bookings(customer_id: str) -> list[Booking]:
    db = _get_db()
    doc = await db.collection("customers").document(customer_id).get()
    if not doc.exists:
        return []
    data = doc.to_dict()
    pnrs = data.get("bookings", [])
    bookings = []
    for pnr in pnrs:
        booking = await get_booking(pnr)
        if booking:
            bookings.append(booking)
    return bookings
