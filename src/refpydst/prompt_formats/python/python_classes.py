import abc
from dataclasses import dataclass
from typing import Literal, Union

PriceRange = Literal["dontcare", "cheap", "moderate", "expensive"]
HotelType = Literal["hotel", "guest house", "dontcare"]
Option = Literal["yes", "no", "dontcare"]
DayOfWeek = Literal["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
Area = Literal["dontcare", "centre", "east", "north", "south", "west"]


@dataclass
class Hotel:
    name: str = None
    price_range: PriceRange = None
    type: HotelType = None
    parking: Option = None
    book_number_of_days: int = None
    book_day: DayOfWeek = None
    book_people: int = None
    area: Area = None
    stars: Union[int, Literal["dontcare"]] = None  # between 0 and 5 or dontcare
    internet: Option = None


@dataclass
class Train:
    destination: str = None
    leave_from: str = None
    day: DayOfWeek = None
    book_people: int = None
    depart_time: str = None  # hh:mm or dontcare
    arrive_by_time: str = None  # hh:mm or dontcare


AttractionType = Literal["architecture", "boat", "church", "cinema", "college", "concert hall", "entertainment",
                         "hotspot", "multiple sports", "museum", "nightclub", "park", "special", "swimming pool",
                         "theatre", "dontcare"]


@dataclass
class Attraction:
    name: str = None
    area: Area = None
    type: AttractionType = None


@dataclass
class Restaurant:
    name: str = None
    food_type: str = None
    price_range: PriceRange = None
    area: Area = None
    book_time: str = None  # hh:mm or dontcare
    book_day: DayOfWeek = None
    book_people: int = None


@dataclass
class Taxi:
    destination: str = None
    leave_from: str = None
    depart_time: str = None  # hh:mm or dontcare
    arrive_by_time: str = None  # hh:mm or dontcare


@dataclass
class BeliefState:
    hotel: Hotel = None
    train: Train = None
    attraction: Attraction = None
    restaurant: Restaurant = None
    taxi: Taxi = None


class DialogueAgent(abc.ABC):

    state: BeliefState

    @abc.abstractmethod
    def find_hotel(self, name: str = None, price_range: PriceRange = None, type: HotelType = None,
                    parking: Option = None, book_number_of_days: int = None, book_day: DayOfWeek = None,
                    book_people: int = None, area: Area = None, stars: Union[int, Literal["dontcare"]] = None,
                    internet: Option = None) -> Hotel:
        pass

    @abc.abstractmethod
    def find_train(self, destination: str = None, leave_from: str = None, day: DayOfWeek = None,
                   book_people: int = None, depart_time: str = None, arrive_by_time: str = None) -> Train:
        pass

    @abc.abstractmethod
    def find_attraction(self, name: str = None, area: Area = None, type: AttractionType = None) -> Attraction:
        pass

    @abc.abstractmethod
    def find_restaurant(self, name: str = None, food_type: str = None, price_range: PriceRange = None,
                        area: Area = None, book_time: str = None, book_day: DayOfWeek = None,
                        book_people: int = None, ) -> Restaurant:
        pass

    @abc.abstractmethod
    def find_taxi(self, destination: str = None, leave_from: str = None, depart_time: str = None,
                  arrive_by_time: str = None) -> Taxi:
        pass

    def get_state(self) -> BeliefState:
        return self.state


if __name__ == '__main__':
    agent = DialogueAgent()
    state = BeliefState()

