from __future__ import annotations

import json
import logging
import os

import httpx
from livekit.agents import Agent, RunContext, function_tool

import firestore_client
from models import (
    Booking,
    BookingStatus,
    CabinClass,
    FareBreakdown,
    FlightOption,
    FlightSegment,
    Passenger,
    UserData,
)

logger = logging.getLogger("zara")

# Shared voice configuration — these voice names work in both
# google.TTS (Gemini TTS) and google.realtime.RealtimeModel (Live API).
VOICE_NAME = "Despina"

# Pipeline TTS controls (google.TTS only — realtime has no equivalent)
SPEAKING_RATE = 0.8  # 0.25 to 4.0, where 1.0 is normal speed
PITCH = -5  # -20 to +20 semitones relative to the original pitch

SPEECH_STYLE = (
    "Support both Urdu and English naturally. "
    "Speak in a natural Pakistani call-center style. "
    "Use a clear, conversational tone with a moderate pace, "
    "slightly low pitch - not slow or robotic. "
    "Keep sentences flowing smoothly with short natural pauses, "
    "like a real helpdesk agent speaking to a customer. "
    "Sound polite, confident, and professional. "
    "Do not exaggerate pauses or stretch words. "
    "Maintain a natural rhythm similar to everyday conversation. "
    "Always spell out numbers and emails in English."
)

ZARA_INSTRUCTIONS = """\
# Identity

You are Zara, a female helpdesk assistant for a travel agency, \
specializing in air travel. You speak both Urdu and English.

You help customers with account authentication, booking lookups, \
cancellations, and routing to the booking team. You only handle air travel. \
If asked about hotels, car rentals, or anything outside air travel, \
politely say that is not something you handle.

# Output rules

Respond in plain text only.\
Keep replies to one to three short sentences. Ask only one question at a time.
Spell out PNR codes letter by letter.
Spell out email addresses clearly.
Say amounts in words, like fifteen thousand five hundred seventy five rupees.

# Language

Support both English and Urdu naturally. Match the language the customer \
uses. Mix English and Urdu as is natural in Pakistani conversation.

# Conversational style

Sound like a professional Pakistani call center agent. Be polite, \
confident, and warm.
Infer context from conversation. If the customer says me and my wife, \
that means 2 passengers. If they say next Friday, work out the date. \
If they say direct flight only, note that preference.
When a customer says they have used your service before or have an \
account, ask for their name and PIN to authenticate.

# Routing

When the customer wants to search for flights, book a flight, or anything \
related to making a new booking, transfer them to the booking team.

# Tools

Use your tools to authenticate customers, look up bookings, and cancel bookings.
When a tool returns an error, tell the customer there is a temporary issue \
and suggest trying again later. Never retry the same tool call silently.

# Guardrails

Never fabricate flight information, prices, or PNR codes. Only use data \
from your tools.
Do not help with anything outside air travel bookings.
Protect customer PINs. Never read back a PIN. Only confirm it was set.
If authentication fails, allow up to two retries then suggest they \
continue as a new customer.\
"""

BOOKING_INSTRUCTIONS = """\
# Identity

You are Zara's booking specialist. You handle flight search, fare details, \
booking, ticketing, and account creation for a Pakistani travel agency. \
You speak both Urdu and English.

# Output rules

Respond in plain text only. Never use markdown, bullet points, numbered \
lists, asterisks, emojis, or special formatting.
Keep replies to one to three short sentences. Ask only one question at a time.
Spell out PNR codes letter by letter, for example P as in Papa, K as in Kilo.
Spell out email addresses clearly.
Say amounts in words, like fifteen thousand five hundred seventy five rupees.
Never say JSON, parameters, function names, or technical terms.

# Language

Support both English and Urdu naturally. Match the language the customer \
uses. Mix English and Urdu as is natural in Pakistani conversation.

# Conversational style

Sound like a professional Pakistani call center agent. Be polite, \
confident, and warm.
Infer context from conversation. If the customer says me and my wife, \
that means 2 passengers. If they say next Friday, work out the date. \
If they say direct flight only, note that preference.
Assume defaults unless told otherwise. Default is 1 adult passenger in \
economy class. Do not list all options like a robot.

# Booking flow

Always search for flights before offering options. Never make up flight \
numbers or prices.
Before booking, you must collect the passenger's full name, date of birth, \
gender, passport number, email address, and phone number. Never use \
placeholder values. If any detail is missing, ask for it.
After booking, always issue the ticket and send confirmation unless the \
customer says otherwise.

# After booking and ticketing

You must ask the customer to create a 4-digit PIN for their account.
PIN creation is required. Explain that without a PIN they will not be \
able to inquire about their booking or get assistance in future calls an.
If the customer refuses after your explanation, accept their decision \
but clearly warn them one final time about the limitation.
After PIN creation or if the customer declines, transfer back to the \
main agent.

# Tools

Use your tools to search flights, check fares, book flights, issue \
tickets, and create PINs.
When a tool returns an error, tell the customer there is a temporary issue \
and suggest trying again later. Never retry the same tool call silently.

# Guardrails

Never fabricate flight information, prices, or PNR codes. Only use data \
from your tools.
Protect customer PINs. Never read back a PIN. Only confirm it was set.\
"""


class Zara(Agent):
    def __init__(self, **kwargs) -> None:
        self._returning = "chat_ctx" in kwargs
        super().__init__(instructions=ZARA_INSTRUCTIONS, **kwargs)

    async def on_enter(self) -> None:
        if self._returning:
            await self.session.generate_reply(
                instructions=(
                    "The customer has been transferred back to you after completing "
                    "their booking. Ask if there is anything else you can help with."
                )
            )
        else:
            await self.session.generate_reply(
                instructions=(
                    "You are initiating the conversation. The customer has not spoken yet. "
                    "Say exactly: Assalamualaikum, Aap ki baat Zara sy horahi hai, "
                    "how can I help you today? Do not say Walaikum Asalam — that is "
                    "a reply, not an opening greeting."
                )
            )

    @function_tool()
    async def transfer_to_booking(self, context: RunContext):
        """Transfer to the booking specialist when the customer wants to search or book flights."""
        return BookingAgent(chat_ctx=self.chat_ctx), "Transferring to booking"

    @function_tool()
    async def authenticate_customer(
        self,
        context: RunContext[UserData],
        name: str,
        pin: str,
    ) -> str:
        """Authenticate a returning customer using their name and PIN.

        Args:
            name: The customer's name
            pin: The customer's 4-digit PIN
        """
        try:
            profile = await firestore_client.authenticate_customer(name, pin)
        except Exception:
            logger.exception("Authentication failed")
            return "I am sorry, I could not verify your account right now. Please try again."

        if not profile:
            return (
                "I could not verify those details. "
                "Please check your name and PIN and try again."
            )

        context.userdata.is_authenticated = True
        context.userdata.customer_profile = profile

        # Inject customer context
        bookings = await firestore_client.get_customer_bookings(profile.customer_id)
        booking_summary = ", ".join(b.pnr for b in bookings) if bookings else "none"
        chat_ctx = self.chat_ctx.copy()
        chat_ctx.add_message(
            role="system",
            content=(
                f"Customer authenticated. Name: {profile.name}. "
                f"Email: {profile.email}. Phone: {profile.phone}. "
                f"Previous bookings: {booking_summary}. "
                f"Personalize all responses."
            ),
        )
        await self.update_chat_ctx(chat_ctx)

        return (
            f"Welcome back, {profile.name}. You are now verified. "
            f"How can I help you today?"
        )

    @function_tool()
    async def lookup_booking(
        self,
        context: RunContext[UserData],
        pnr: str,
    ) -> str:
        """Look up details of an existing booking.

        Args:
            pnr: The 6-character PNR code to look up
        """
        try:
            booking = await firestore_client.get_booking(pnr)
        except Exception:
            logger.exception("Failed to look up booking")
            return "I am sorry, I could not look up that booking right now. Please try again."

        if not booking:
            return f"I could not find a booking with PNR {pnr}."

        pax_names = ", ".join(
            f"{p.title} {p.first_name} {p.last_name}" for p in booking.passengers
        )
        return (
            f"Booking {booking.pnr}: {booking.flight.airline} flight "
            f"{booking.flight.flight_number} from {booking.flight.origin} to "
            f"{booking.flight.destination}, departing {booking.flight.departure_time}. "
            f"Status {booking.status.value}. "
            f"Passengers: {pax_names}. "
            f"Total fare {booking.fare.total} rupees."
        )

    @function_tool()
    async def cancel_booking(
        self,
        context: RunContext[UserData],
        pnr: str,
    ) -> str:
        """Cancel an existing booking.

        Args:
            pnr: The 6-character PNR code of the booking to cancel
        """
        context.disallow_interruptions()

        try:
            booking = await firestore_client.update_booking_status(
                pnr, BookingStatus.CANCELLED
            )
        except Exception:
            logger.exception("Failed to cancel booking")
            return "I am sorry, there was an error cancelling the booking. Please try again."

        if not booking:
            return f"I could not find a booking with PNR {pnr}."

        return f"Booking {pnr} has been cancelled successfully."


class BookingAgent(Agent):
    def __init__(self, **kwargs) -> None:
        self._is_handoff = "chat_ctx" in kwargs
        super().__init__(instructions=BOOKING_INSTRUCTIONS, **kwargs)

    async def on_enter(self) -> None:
        if self._is_handoff:
            await self.session.generate_reply(
                instructions=(
                    "The customer has been transferred to you for booking assistance. "
                    "Review the conversation context and continue helping them. "
                    "Do not re-introduce yourself or repeat a greeting."
                )
            )

    @function_tool()
    async def search_flights(
        self,
        context: RunContext[UserData],
        origin: str,
        destination: str,
        travel_date: str,
        cabin_class: str = "economy",
        num_passengers: int = 1,
    ) -> str:
        """Search for available flights between two cities.

        Args:
            origin: Origin city or location name as spoken by the customer
            destination: Destination city or location name as spoken by the customer
            travel_date: Travel date in YYYY-MM-DD format
            cabin_class: Cabin class preference (economy, premium_economy, business, first)
            num_passengers: Number of passengers
        """
        webhook_url = os.getenv("N8N_FLIGHTS_WEBHOOK_URL")
        if not webhook_url:
            return "Flight search is temporarily unavailable. Please try again later."

        payload = {
            "origin": origin,
            "destination": destination,
            "travel_date": travel_date,
            "cabin_class": cabin_class,
            "num_passengers": num_passengers,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(webhook_url, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            logger.exception("Flight search webhook failed")
            return (
                "I am sorry, I could not reach the flight search service right now. "
                "Please try again in a moment."
            )

        flights = data.get("flights", [])
        if not flights:
            return (
                f"I could not find any flights from {origin} to {destination} "
                f"on {travel_date}. Would you like to try a different date?"
            )

        options: list[FlightOption] = []
        for i, f in enumerate(flights, 1):
            fare_data = f.get("fare", {})
            cabin = CabinClass(fare_data.get("cabin_class", cabin_class))
            option = FlightOption(
                option_id=f"OPT-{i}",
                segment=FlightSegment(
                    flight_number=f["flight_number"],
                    airline=f["airline"],
                    origin=f["origin"],
                    destination=f["destination"],
                    departure_time=f["departure_time"],
                    arrival_time=f["arrival_time"],
                    duration_minutes=f["duration_minutes"],
                    aircraft=f.get("aircraft", ""),
                ),
                fare=FareBreakdown(
                    base_fare=fare_data["base_fare"],
                    taxes=fare_data["taxes"],
                    fuel_surcharge=fare_data["fuel_surcharge"],
                    total=fare_data["total"],
                    cabin_class=cabin,
                ),
                seats_available=f.get("seats_available", 0),
            )
            options.append(option)

        context.userdata.flight_options = options

        # Build readable summary
        lines = []
        for opt in options:
            s = opt.segment
            lines.append(
                f"Option {opt.option_id}: {s.airline} flight {s.flight_number}, "
                f"departing {s.departure_time} arriving {s.arrival_time}, "
                f"duration {s.duration_minutes} minutes, "
                f"total fare {opt.fare.total} rupees, "
                f"{opt.seats_available} seats available"
            )
        return ". ".join(lines)

    @function_tool()
    async def get_fare_details(
        self,
        context: RunContext[UserData],
        option_number: int,
    ) -> str:
        """Get detailed fare breakdown for a flight option.

        Args:
            option_number: The option number from the search results (1, 2, 3, etc.)
        """
        options = context.userdata.flight_options
        if not options:
            return (
                "No flight search results available. Please search for flights first."
            )

        idx = option_number - 1
        if idx < 0 or idx >= len(options):
            return f"Invalid option number. Please choose between 1 and {len(options)}."

        opt = options[idx]
        fare = opt.fare
        return (
            f"Fare breakdown for {opt.segment.airline} flight "
            f"{opt.segment.flight_number}: "
            f"Base fare {fare.base_fare} rupees, "
            f"taxes {fare.taxes} rupees, "
            f"fuel surcharge {fare.fuel_surcharge} rupees, "
            f"total {fare.total} rupees in {fare.cabin_class.value} class"
        )

    @function_tool()
    async def book_flight(
        self,
        context: RunContext[UserData],
        option_number: int,
        passengers: str,
        contact_email: str,
        contact_phone: str,
    ) -> str:
        """Book a flight for the given passengers. IMPORTANT: You must collect all passenger details and contact information from the customer BEFORE calling this tool. Never use placeholder values like not_provided or unknown. If any required information is missing, ask the customer for it instead of calling this tool.

        Args:
            option_number: The option number from the search results (1, 2, 3, etc.)
            passengers: JSON array of passenger objects. Each must have real values for: title, first_name, last_name, date_of_birth (YYYY-MM-DD), gender, passport_number, contact_phone
            contact_email: The customer's real email address collected from conversation
            contact_phone: The customer's real phone number collected from conversation
        """
        context.disallow_interruptions()

        options = context.userdata.flight_options
        if not options:
            return (
                "No flight search results available. Please search for flights first."
            )

        idx = option_number - 1
        if idx < 0 or idx >= len(options):
            return f"Invalid option number. Please choose between 1 and {len(options)}."

        # Reject placeholder values
        placeholders = {"not_provided", "unknown", "n/a", "none", "null", ""}
        if contact_email.strip().lower() in placeholders:
            return "I need the customer's email address before I can book. Please ask them."
        if contact_phone.strip().lower() in placeholders:
            return (
                "I need the customer's phone number before I can book. Please ask them."
            )

        try:
            pax_list = json.loads(passengers)
        except json.JSONDecodeError:
            return "I could not process the passenger details. Please try again."

        if not pax_list:
            return "No passenger details provided. Please collect passenger information first."

        for p in pax_list:
            name = f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
            if not name or name.lower() in placeholders:
                return "I need each passenger's real name before booking. Please ask the customer."

        pax_objects = [
            Passenger(
                title=p.get("title", ""),
                first_name=p.get("first_name", ""),
                last_name=p.get("last_name", ""),
                date_of_birth=p.get("date_of_birth", ""),
                gender=p.get("gender", ""),
                passport_number=p.get("passport_number", ""),
                contact_phone=p.get("contact_phone", ""),
            )
            for p in pax_list
        ]

        opt = options[idx]
        booking = Booking(
            pnr="",  # will be generated by firestore_client
            flight=opt.segment,
            fare=opt.fare,
            passengers=pax_objects,
            contact_email=contact_email,
            contact_phone=contact_phone,
            status=BookingStatus.CONFIRMED,
            created_at="",  # will be set by firestore_client
        )

        try:
            await firestore_client.save_booking(booking)
        except Exception:
            logger.exception("Failed to save booking")
            return "There is a system issue preventing the booking right now. Do not retry automatically. Inform the customer and ask them to try again later."

        context.userdata.current_pnr = booking.pnr
        return (
            f"Booking confirmed. Your PNR is {booking.pnr}. "
            f"Flight {opt.segment.flight_number} from {opt.segment.origin} to "
            f"{opt.segment.destination} on {opt.segment.departure_time}. "
            f"Total fare {opt.fare.total} rupees for {len(pax_objects)} passenger(s)."
        )

    @function_tool()
    async def issue_ticket(
        self,
        context: RunContext[UserData],
        pnr: str,
    ) -> str:
        """Issue a ticket for an existing confirmed booking.

        Args:
            pnr: The 6-character PNR code of the booking
        """
        context.disallow_interruptions()

        try:
            booking = await firestore_client.update_booking_status(
                pnr, BookingStatus.TICKETED
            )
        except Exception:
            logger.exception("Failed to issue ticket")
            return (
                "I am sorry, there was an error issuing the ticket. Please try again."
            )

        if not booking:
            return f"I could not find a booking with PNR {pnr}."

        # Stub: log confirmation email (real sending added later)
        logger.info(
            "Confirmation email would be sent to %s for PNR %s",
            booking.contact_email,
            pnr,
        )

        return (
            f"Ticket issued successfully for PNR {pnr}. "
            f"A confirmation email will be sent to {booking.contact_email}."
        )

    @function_tool()
    async def create_customer_pin(
        self,
        context: RunContext[UserData],
        name: str,
        email: str,
        phone: str,
        pin: str,
    ) -> str:
        """Create a new customer account with a 4-digit PIN.

        Args:
            name: The customer's full name
            email: The customer's email address
            phone: The customer's phone number
            pin: A 4-digit PIN chosen by the customer
        """
        if not pin.isdigit() or len(pin) != 4:
            return "The PIN must be exactly 4 digits. Please choose a valid PIN."

        pnr = context.userdata.current_pnr

        try:
            profile = await firestore_client.create_customer(
                name=name, email=email, phone=phone, pin=pin, pnr=pnr
            )
        except Exception:
            logger.exception("Failed to create customer")
            return "I am sorry, I could not create your account right now. Please try again."

        context.userdata.is_authenticated = True
        context.userdata.customer_profile = profile

        return (
            "Your account has been created and your PIN has been set successfully. "
            "You can use your name and PIN to access your bookings in future calls."
        )

    @function_tool()
    async def transfer_to_main(self, context: RunContext):
        """Transfer customer back to the main agent after booking is complete."""
        return Zara(chat_ctx=self.chat_ctx), "Transferring back to main agent"
