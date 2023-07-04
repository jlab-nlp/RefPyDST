# DB/Ontology Management

The primary purpose of this sub-module is to support the Ontology class, which reads entities from a database and 
supports a `get_canonical` method, which transforms a slot-name and slot-value string in to the canonical form for that
entity, if it exists in the database. This handles minor typos, appended or removed articles, etc.

See db.test_ontology.OntologyTests.test_simple_get_canonical for an example.

## Database files

Database files are stored in the [multiwoz sub-directory](./multiwoz), sourced from original repo. See details there.