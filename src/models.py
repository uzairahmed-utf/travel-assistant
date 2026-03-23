from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class CabinClass(Enum):
    ECONOMY = "economy"
    PREMIUM_ECONOMY = "premium_economy"
    BUSINESS = "business"
    FIRST = "first"


class BookingStatus(Enum):
    CONFIRMED = "confirmed"
    TICKETED = "ticketed"
    CANCELLED = "cancelled"


@dataclass
class FlightSegment:
    flight_number: str
    airline: str
    origin: str
    destination: str
    departure_time: str
    arrival_time: str
    duration_minutes: int
    aircraft: str


@dataclass
class FareBreakdown:
    base_fare: int
    taxes: int
    fuel_surcharge: int
    total: int
    cabin_class: CabinClass


@dataclass
class FlightOption:
    option_id: str
    segment: FlightSegment
    fare: FareBreakdown
    seats_available: int


@dataclass
class Booking:
    pnr: str
    customer_id: str
    flight: FlightSegment
    fare: FareBreakdown
    status: BookingStatus
    created_at: str


@dataclass
class CustomerProfile:
    customer_id: str
    name: str
    pin: str
    email: str
    phone: str
    date_of_birth: str = ""
    gender: str = ""
    passport_number: str = ""
    bookings: list[str] = field(default_factory=list)


@dataclass
class UserData:
    is_authenticated: bool = False
    customer_profile: CustomerProfile | None = None
    flight_options: list[FlightOption] = field(default_factory=list)
    current_pnr: str | None = None
    _booking_details: dict = field(default_factory=dict)
