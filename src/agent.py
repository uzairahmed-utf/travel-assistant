import logging
import os

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    cli,
    room_io,
)
from livekit.plugins import google, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from assistant import PITCH, SPEAKING_RATE, SPEECH_STYLE, VOICE_NAME, Assistant

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# Select agent mode: "pipeline" (STT→LLM→TTS) or "realtime" (Gemini Live API)
AGENT_MODE = os.getenv("AGENT_MODE", "pipeline")

server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


def _room_options():
    """Shared room options with noise cancellation for both entry points."""
    return room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(
            noise_cancellation=lambda params: (
                noise_cancellation.BVCTelephony()
                if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                else noise_cancellation.BVC()
            ),
        ),
    )


if AGENT_MODE == "realtime":

    @server.rtc_session(agent_name="realtime")
    async def realtime_agent(ctx: JobContext):
        """Gemini Live API — single realtime model handles STT, LLM, and TTS."""
        ctx.log_context_fields = {"room": ctx.room.name}

        session = AgentSession(
            llm=google.realtime.RealtimeModel(
                model="gemini-live-2.5-flash-native-audio",
                voice=VOICE_NAME,
                vertexai=True,
                project="receiptflow-00",
                location="europe-west4",
            ),
            vad=ctx.proc.userdata["vad"],
        )

        await session.start(
            agent=Assistant(),
            room=ctx.room,
            room_options=_room_options(),
        )
        await ctx.connect()

else:

    @server.rtc_session(agent_name="pipeline")
    async def pipeline_agent(ctx: JobContext):
        """Traditional STT → LLM → TTS voice pipeline using Google Cloud services."""
        ctx.log_context_fields = {"room": ctx.room.name}

        session = AgentSession(
            stt=google.STT(
                model="chirp_3",
                location="eu",
                languages=["en-US", "ur-PK"],
                detect_language=True,
                interim_results=True,
                min_confidence_threshold=0.0,
            ),
            llm=google.LLM(
                model="gemini-2.5-flash-lite",
                vertexai=True,
                project="receiptflow-00",
                location="europe-west4",
            ),
            tts=google.TTS(
                model_name="gemini-2.5-flash-tts",
                voice_name=VOICE_NAME,
                speaking_rate=SPEAKING_RATE,
                pitch=PITCH,
                prompt=SPEECH_STYLE,
            ),
            turn_handling=TurnHandlingOptions(
                turn_detection=MultilingualModel(),
            ),
            vad=ctx.proc.userdata["vad"],
            preemptive_generation=True,
        )

        await session.start(
            agent=Assistant(),
            room=ctx.room,
            room_options=_room_options(),
        )
        await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
