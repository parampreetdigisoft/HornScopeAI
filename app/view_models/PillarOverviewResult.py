from typing import List

from pydantic import BaseModel, Field, field_validator


class PillarOverviewCard(BaseModel):
    pillarId: int = Field(..., ge=1)
    pillarName: str = Field(default="", max_length=200)
    summary: str = Field(..., min_length=40, max_length=2500)
    areasForImprovement: str = Field(..., min_length=40, max_length=1500)

    @field_validator("pillarName", "summary", "areasForImprovement")
    @classmethod
    def collapse_prose(cls, value: str) -> str:
        return " ".join(value.split())


class PillarOverviewResult(BaseModel):
    pillars: List[PillarOverviewCard] = Field(..., min_length=1)


class ChatPillarOverviewResponse(BaseModel):
    success: bool
    message: str
    result: PillarOverviewResult
