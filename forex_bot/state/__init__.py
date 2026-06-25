"""Durable state so the bot is restartable without losing what it knows."""

from .store import StateStore

__all__ = ["StateStore"]
