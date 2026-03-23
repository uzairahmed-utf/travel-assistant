import json

import pytest
from livekit.agents import AgentSession, inference, llm, mock_tools

from assistant import BookingAgent, Zara
from models import UserData


def _llm() -> llm.LLM:
    return inference.LLM(model="openai/gpt-4.1-mini")


def _session(**kwargs) -> AgentSession:
    return AgentSession(userdata=UserData(), **kwargs)


# --- A. Greeting & persona ---


@pytest.mark.asyncio
async def test_greeting() -> None:
    """Zara greets in Urdu and introduces herself by name."""
    async with (
        _llm() as llm,
        _session(llm=llm) as session,
    ):
        await session.start(Zara())

        result = await session.run(user_input="Hello")

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent=(
                    "Responds warmly and professionally as a travel assistant. "
                    "May use Urdu greeting like Assalamualaikum. "
                    "Offers help with air travel."
                ),
            )
        )


@pytest.mark.asyncio
async def test_out_of_scope_refusal() -> None:
    """Zara refuses non-air-travel requests politely."""
    async with (
        _llm() as llm,
        _session(llm=llm) as session,
    ):
        await session.start(Zara())

        result = await session.run(user_input="Book me a hotel in Lahore")

        await (
            result.expect.next_event()
            .is_message(role="assistant")
            .judge(
                llm,
                intent=(
                    "Politely declines or refuses the hotel booking request. "
                    "Explains that they only handle air travel. "
                    "Does not attempt to book a hotel."
                ),
            )
        )
        result.expect.no_more_events()


# --- B. Handoffs ---


@pytest.mark.asyncio
async def test_handoff_to_booking() -> None:
    """Zara transfers to BookingAgent when user wants to book flights."""
    async with (
        _llm() as llm,
        _session(llm=llm) as session,
    ):
        await session.start(Zara())

        result = await session.run(
            user_input="I want to search for flights from Karachi to Lahore"
        )

        result.expect.contains_agent_handoff(new_agent_type=BookingAgent)


@pytest.mark.asyncio
async def test_handoff_back_to_main() -> None:
    """BookingAgent transfers back to Zara when booking flow is done."""
    async with (
        _llm() as llm,
        _session(llm=llm) as session,
    ):
        await session.start(BookingAgent())

        result = await session.run(
            user_input=(
                "Everything is done, my booking is complete and PIN is set. "
                "Please transfer me back to the main agent."
            )
        )

        result.expect.contains_agent_handoff(new_agent_type=Zara)


# --- C. Flight search ---


def _mock_search_flights(
    origin: str,
    destination: str,
    travel_date: str,
    cabin_class: str = "economy",
) -> str:
    return (
        "Option OPT-1: PIA flight PK-303, departing 08:30 arriving 10:15, "
        "duration 105 minutes, total fare 16575 rupees, 24 seats available. "
        "Option OPT-2: Airblue flight PA-200, departing 14:00 arriving 15:45, "
        "duration 105 minutes, total fare 14200 rupees, 12 seats available"
    )


@pytest.mark.asyncio
async def test_search_flights_tool_call() -> None:
    """Searching flights calls the search_flights tool and presents options."""
    async with (
        _llm() as llm,
        _session(llm=llm) as session,
    ):
        await session.start(BookingAgent())

        with mock_tools(BookingAgent, {"search_flights": _mock_search_flights}):
            result = await session.run(
                user_input=(
                    "I want to fly from Karachi to Lahore on 15 April 2026, "
                    "one passenger economy class"
                )
            )

            result.expect.next_event().is_function_call(name="search_flights")
            result.expect.next_event().is_function_call_output()

            await (
                result.expect.next_event()
                .is_message(role="assistant")
                .judge(
                    llm,
                    intent=(
                        "Presents flight options to the customer. "
                        "Mentions at least one flight with its details "
                        "such as airline, timing, and fare."
                    ),
                )
            )


@pytest.mark.asyncio
async def test_search_infers_defaults() -> None:
    """Search defaults to 1 passenger in economy when not specified."""
    async with (
        _llm() as llm,
        _session(llm=llm) as session,
    ):
        await session.start(BookingAgent())

        with mock_tools(BookingAgent, {"search_flights": _mock_search_flights}):
            result = await session.run(
                user_input="Search flights from Islamabad to Karachi on 20 April 2026"
            )

            fnc = result.expect.next_event().is_function_call(name="search_flights")
            raw_args = fnc.event().item.arguments
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            assert args.get("cabin_class", "economy") == "economy"


# --- D. Fare details ---


def _mock_get_fare_details(option_number: int) -> str:
    return (
        "Fare breakdown for PIA flight PK-303: "
        "Base fare 12500 rupees, taxes 1875 rupees, "
        "fuel surcharge 2200 rupees, total 16575 rupees in economy class"
    )


@pytest.mark.asyncio
async def test_fare_breakdown() -> None:
    """Multi-turn: search then ask for fare details."""
    async with (
        _llm() as llm,
        _session(llm=llm) as session,
    ):
        await session.start(BookingAgent())

        with mock_tools(
            BookingAgent,
            {
                "search_flights": _mock_search_flights,
                "get_fare_details": _mock_get_fare_details,
            },
        ):
            # Turn 1: search
            await session.run(
                user_input="Search flights Karachi to Lahore on 15 April 2026, one passenger economy"
            )

            # Turn 2: ask for fare
            result = await session.run(
                user_input="What is the fare breakdown for option 1?"
            )

            result.expect.next_event().is_function_call(name="get_fare_details")
            result.expect.next_event().is_function_call_output()

            await (
                result.expect.next_event()
                .is_message(role="assistant")
                .judge(
                    llm,
                    intent=(
                        "Provides a fare breakdown including base fare, "
                        "taxes, fuel surcharge, and total amount in rupees."
                    ),
                )
            )


# --- E. Booking ---


def _mock_book_flight(
    option_number: int,
    passengers: str,
    contact_email: str,
    contact_phone: str,
) -> str:
    return (
        "Booking confirmed. Your PNR is AB3K9X. "
        "Flight PK-303 from KHI to LHE on 08:30. "
        "Total fare 16575 rupees for 1 passenger(s)."
    )


@pytest.mark.asyncio
async def test_book_flight() -> None:
    """Booking a flight calls book_flight and returns PNR."""
    async with (
        _llm() as llm,
        _session(llm=llm) as session,
    ):
        await session.start(BookingAgent())

        with mock_tools(
            BookingAgent,
            {
                "search_flights": _mock_search_flights,
                "book_flight": _mock_book_flight,
            },
        ):
            # Turn 1: search
            await session.run(
                user_input="Search flights Karachi to Lahore on 15 April 2026, one passenger economy"
            )

            # Turn 2: book with all details provided explicitly
            result = await session.run(
                user_input=(
                    "I confirm, please book option 1. Passenger details: "
                    "Mr Ahmed Khan, date of birth 1990-05-15, male, "
                    "passport number AB1234567, phone 03001234567. "
                    "Contact email is ahmed@example.com and contact phone is 03001234567."
                )
            )

            # Agent must call book_flight somewhere in this turn
            result.expect.contains_function_call(name="book_flight")


def _mock_issue_ticket(pnr: str) -> str:
    return (
        "Ticket issued successfully for PNR AB3K9X. "
        "A confirmation email will be sent to ahmed@example.com."
    )


@pytest.mark.asyncio
async def test_issue_ticket_after_booking() -> None:
    """Issue ticket calls issue_ticket tool."""
    async with (
        _llm() as llm,
        _session(llm=llm) as session,
    ):
        await session.start(BookingAgent())

        with mock_tools(
            BookingAgent,
            {
                "search_flights": _mock_search_flights,
                "book_flight": _mock_book_flight,
                "issue_ticket": _mock_issue_ticket,
            },
        ):
            # Turn 1: search
            await session.run(
                user_input="Search flights Karachi to Lahore on 15 April 2026, one passenger economy"
            )
            # Turn 2: book with all details and request ticket issuance
            result = await session.run(
                user_input=(
                    "I confirm, book option 1. Passenger: Mr Ahmed Khan, "
                    "born 1990-05-15, male, passport AB1234567, phone 03001234567. "
                    "Email ahmed@example.com, phone 03001234567. "
                    "Please issue the ticket as well."
                )
            )

            result.expect.contains_function_call(name="issue_ticket")


# --- F. Cancellation ---


def _mock_cancel_booking(pnr: str) -> str:
    return "Booking ABC123 has been cancelled successfully."


@pytest.mark.asyncio
async def test_cancel_booking() -> None:
    """Cancellation calls cancel_booking tool after confirmation."""
    async with (
        _llm() as llm,
        _session(llm=llm) as session,
    ):
        await session.start(Zara())

        with mock_tools(Zara, {"cancel_booking": _mock_cancel_booking}):
            # Turn 1: request cancellation — agent may ask for auth or confirmation
            await session.run(
                user_input="I need to cancel my booking, PNR is ABC123."
            )

            # Turn 2: confirm without authentication
            result = await session.run(
                user_input=(
                    "I do not have an account. I confirm I want to cancel "
                    "booking ABC123. Please go ahead."
                )
            )

            result.expect.contains_function_call(name="cancel_booking")


# --- G. Authentication ---


def _mock_auth_success(name: str, pin: str) -> str:
    return "Welcome back, Ahmed Khan. You are now verified. How can I help you today?"


def _mock_auth_failure(name: str, pin: str) -> str:
    return (
        "I could not verify those details. "
        "Please check your name and PIN and try again."
    )


@pytest.mark.asyncio
async def test_authenticate_success() -> None:
    """Successful authentication calls authenticate_customer and personalizes."""
    async with (
        _llm() as llm,
        _session(llm=llm) as session,
    ):
        await session.start(Zara())

        with mock_tools(Zara, {"authenticate_customer": _mock_auth_success}):
            result = await session.run(
                user_input=(
                    "I have an account. My name is Ahmed Khan and my PIN is 1234."
                )
            )

            result.expect.contains_function_call(name="authenticate_customer")


@pytest.mark.asyncio
async def test_authenticate_failure() -> None:
    """Failed authentication returns a graceful message."""
    async with (
        _llm() as llm,
        _session(llm=llm) as session,
    ):
        await session.start(Zara())

        with mock_tools(Zara, {"authenticate_customer": _mock_auth_failure}):
            result = await session.run(user_input="My name is Ahmed Khan, PIN 9999")

            result.expect.contains_function_call(name="authenticate_customer")

            # The last message should inform about failure (LLM may speak
            # before the tool call too, so use the last event)
            await (
                result.expect[-1]
                .is_message(role="assistant")
                .judge(
                    llm,
                    intent=(
                        "Tells the customer that verification failed and "
                        "suggests they check their details or try again. "
                        "May respond in English, Urdu, or a mix of both. "
                        "Does not show technical errors."
                    ),
                )
            )


# --- H. PIN creation & bilingual ---


def _mock_create_pin(name: str, email: str, phone: str, pin: str) -> str:
    return (
        "Your account has been created and your PIN has been set successfully. "
        "You can use your name and PIN to access your bookings in future calls."
    )


@pytest.mark.asyncio
async def test_create_pin() -> None:
    """PIN creation calls create_customer_pin tool."""
    async with (
        _llm() as llm,
        _session(llm=llm) as session,
    ):
        await session.start(BookingAgent())

        with mock_tools(
            BookingAgent,
            {
                "search_flights": _mock_search_flights,
                "book_flight": _mock_book_flight,
                "issue_ticket": _mock_issue_ticket,
                "create_customer_pin": _mock_create_pin,
            },
        ):
            # Turn 1: search
            await session.run(
                user_input="Search flights Karachi to Lahore on 15 April 2026, one passenger economy"
            )
            # Turn 2: book with all details
            await session.run(
                user_input=(
                    "I confirm, book option 1. Passenger: Mr Ahmed Khan, "
                    "born 1990-05-15, male, passport AB1234567, phone 03001234567. "
                    "Email ahmed@example.com, phone 03001234567."
                )
            )
            # Turn 3: confirm and issue ticket
            await session.run(user_input="Yes confirmed, issue the ticket now")
            # Turn 4: create PIN
            result = await session.run(
                user_input=(
                    "Yes, create my account. Name Ahmed Khan, email ahmed@example.com, "
                    "phone 03001234567, PIN 5678."
                )
            )

            result.expect.contains_function_call(name="create_customer_pin")


@pytest.mark.asyncio
async def test_bilingual_urdu_english() -> None:
    """BookingAgent handles mixed Urdu/English input helpfully."""
    async with (
        _llm() as llm,
        _session(llm=llm) as session,
    ):
        await session.start(BookingAgent())

        with mock_tools(BookingAgent, {"search_flights": _mock_search_flights}):
            result = await session.run(
                user_input="Mujhe Karachi se Lahore jaana hai, 15 April ko flight dekhein"
            )

            await result.expect.next_event(type="message").judge(
                llm,
                intent=(
                    "Responds helpfully to a bilingual Urdu/English request about "
                    "flights from Karachi to Lahore. Does not express confusion "
                    "about the mixed language. May call search_flights or ask a "
                    "clarifying question."
                ),
            )
