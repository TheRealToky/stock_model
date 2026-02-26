from sqlalchemy import (
    Integer,
    Column,
    Text,
    Numeric,
    BigInteger,
    Float,
    Double,
    TIMESTAMP,
    ForeignKey,
    PrimaryKeyConstraint
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import DOUBLE_PRECISION
from db.base import Base


class Company(Base):
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(Text, unique=True, nullable=False)
    company_name = Column(Text)
    market_cap = Column(Numeric)
    country = Column(Text)

    # Relationship to OHLCV
    ohlcv = relationship("OHLCV", back_populates="company")
    clean_ohlcv = relationship("CleanOHLCV", back_populates="company")


class OHLCV(Base):
    __tablename__ = "ohlcv"

    ticker = Column(
        Text,
        ForeignKey("companies.ticker"),
        nullable=False
    )

    timestamp = Column(
        TIMESTAMP(timezone=True),
        nullable=False
    )

    open = Column(DOUBLE_PRECISION)
    high = Column(DOUBLE_PRECISION)
    low = Column(DOUBLE_PRECISION)
    close = Column(DOUBLE_PRECISION)
    volume = Column(BigInteger)
    dividends = Column(Float)
    stock_splits = Column(Float)

    __table_args__ = (
        PrimaryKeyConstraint("ticker", "timestamp"),
    )

    # Relationship back to Company
    company = relationship("Company", back_populates="ohlcv")


# class CleanOHLCV(Base):
#     __tablename__ = "clean_ohlcv"
#
#     ticker = Column(
#         Text,
#         ForeignKey("companies.ticker"),
#         nullable=False
#     )
#
#     timestamp = Column(
#         TIMESTAMP(timezone=True),
#         nullable=False
#     )
#
#     open = Column(DOUBLE_PRECISION)
#     high = Column(DOUBLE_PRECISION)
#     low = Column(DOUBLE_PRECISION)
#     close = Column(DOUBLE_PRECISION)
#     volume = Column(BigInteger)
#     dividends = Column(Float)
#     stock_splits = Column(Float)
#
#     __table_args__ = (
#         PrimaryKeyConstraint("ticker", "timestamp"),
#     )
#
#     # Relationship back to Company
#     company = relationship("Company", back_populates="clean_ohlcv")
