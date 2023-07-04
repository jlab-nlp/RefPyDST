CREATE TABLE hotel(
  name text,
  pricerange text CHECK (pricerange IN (dontcare, cheap, moderate, expensive)),
  type text CHECK (type IN (hotel, guest house)),
  parking text CHECK (parking IN (dontcare, yes, no)),
  book_stay int,
  book_day text,
  book_people int,
  area text CHECK (area IN (dontcare, centre, east, north, south, west)),
  stars int CHECK (stars IN (dontcare, 0, 1, 2, 3, 4, 5)),
  internet text CHECK (internet IN (dontcare, yes, no))
)
/*
4 example rows:
SELECT * FROM hotel LIMIT 4;
name  pricerange  type  parking book_number_of_days book_day  book_people area  stars internet
a and b guest house moderate  guest house  dontcare  3 friday  5 east  4 yes
ashley hotel  expensive hotel yes 2 thursday  5 north 5 yes
el shaddia guest house  cheap guest house  yes 5 friday  2 centre  dontcare  no
express by holiday inn cambridge  dontcare  guest house yes 3 monday  2 east  dontcare  no
*/

CREATE TABLE train(
  destination text,
  departure text,
  day text,
  book_people int,
  leaveat text,
  arriveby text
)
/*
3 example rows:
SELECT * FROM train LIMIT 3;
destination departure day book_people depart_time arrive_by_time
london kings cross  cambridge monday  6 dontcare 05:51
cambridge stansted airport  dontcare  1 20:24 20:52
peterborough  cambridge saturday  2  12:06  12:56
*/

CREATE TABLE attraction(
  name text,
  area text CHECK (area IN (dontcare, centre, east, north, south, west)),
  type text CHECK (type IN (architecture, boat, church, cinema, college, concert hall, entertainment, hotspot, multiple sports, museum, nightclub, park, special, swimming pool, theatre))
)
/*
4 example rows:
SELECT * FROM attraction LIMIT 4;
name area type
abbey pool and astroturf pitch  centre  swimming pool
adc theatre centre  theatre
all saints church dontcare  architecture
castle galleries  centre  museum
*/

CREATE TABLE restaurant(
  name text,
  food text,
  pricerange text CHECK (pricerange IN (dontcare, cheap, moderate, expensive)),
  area text CHECK (area IN (centre, east, north, south, west)),
  book_time text,
  book_day text,
  book_people int
)
/*
5 example rows:
SELECT * FROM restaurant LIMIT 5;
name  food_type  pricerange  area  book_time book_day  book_people
pizza hut city centre italian dontcare centre  13:30 wednesday 7
the missing sock  international moderate  east  dontcare dontcare  2
golden wok chinese moderate north 17:11 friday 4
cambridge chop house  dontcare  expensive  center 08:43 monday  5
darrys cookhouse and wine shop  modern european expensive center  11:20 saturday  8
*/

CREATE TABLE taxi(
  destination text,
  departure text,
  leaveat text,
  arriveby text
)
/*
3 example rows:
SELECT * FROM taxi LIMIT 3;
destination departure depart_time arrive_by_time
copper kettle royal spice 14:45 15:30
magdalene college  university arms hotel dontcare  15:45
lovell lodge  da vinci pizzeria 11:45 dontcare
*/

-- Using valid SQLite, answer the following multi-turn conversational questions for the tables provided above.
