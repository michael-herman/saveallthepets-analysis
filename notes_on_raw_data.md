Notes on Partner Shelter Data Extract
==============

**Author:** *Michael Herman (m.herman.1000@gmail.com)*

# Summary of Data

The initial goal is to create a model that predicts the likely hood of adoption for a specific animal at a given shelter. This data comes from the partner shelter, so it is we can only model the adoption likelihood for this specific location. However, the data schema we establish may be useful for any shelter data. There are a number of tables here that could be used for feature creation. But the main table is _AnimalExtract_, which is unique by specific animal. The _AssesmentExtract_ may be useful for NLP analysis.

The goal of the ETL will be to create a standardized schema limited to useful information in a useful format. At a high level, the ETL will do the following:
- Import limited fields and tables into raw data tables and schemas.
- Process and pull data from the raw data tables into a single table used for modeling. This step would include sub-steps like parsing or summarizing multiple events for an animal to create additional animal-specific features (e.g. number of previous adoptions).

# Overview of Tables

There are 9 different tables included in the data extract. Below are
details on the meaningful tables:
* **AnimalExtract**: This is the primary table of interest, containing details of each animal, including the adoption indicator.
    - The unique identifier for each animal is `ID`.
    - This table can be joined to the _AccountExtract_ table via `ACCOUNT__C`.
    - I identified 44 potentially useful (either for modeling or data manipulation) fields:
        * `ACCOUNT__C`: Used for joining with _AccountExtract_
        * `ADOPTED__C`: Boolean adopted indicator.
        * `ADOPTION_FEE_PAID__C`: Whether the adoption fee was paid, boolean.
        * `ADOPTION_PHASE__C`: The general status of the animal. Values are "Released", "Deceased", "Intake", "Medical to Hold", "Ready to Adopt", "Released", "Schedule Euthanasia", "Temperament", "Veterinary".
        * `AGE_INDICATOR__C`: Units of `AGE__C` column (e.g. "Months" or "Years").
        * `AGE__C`: Integer age value.
        * `ATTITUDE_TOWARDS_PETS__C`: Mostly blank, but non-blanks include "Friendly" or "Skiddish".
        * `ATTITUDES_TOWARDS_KIDS__C`: Mostly blank, but non-blanks include "Friendly" or "Skiddish".
        * `BIRTHDATE__C`: Birthdate of animal. I'm not sure how accurate this is.
        * `COLOR_PATTERN__C`
        * `CREATEDDATE`: The date the animal's record was created. This could be used as an intake date, but not necessarily the date the animal was available for adoption.
        * `DECLAWED_ALL_4__C`: Boolean indicating all paws declawed.
        * `DECLAWED__C`: Boolean indicating declawed (presumably could be not all paws).
        * `DOCKED_TAIL__C`: Boolean indicating docked tail.
        * `EARS_CROPPED_TIPPED__C`: Boolean indicating ears cropped.
        * `GENDER__C`
        * `GENERAL_CONDITION__C`: Mostly blank, but non-blanks include "Good", "Average", "Deplorable".
        * `HOUSETRAINED__C`: Mostly blank but non-blanks are "Yes", "No", or "Unknown".
        * `HOUSETRAINING_NOTES__C`: Text details on housetraining.
        * `ID`: Unique identifier.
        * `IMAGE__C`: Mostly blank, but non-blanks are html links to images. I couldn't get them to pull up, so they may be behind a wall.
        * `LATEST_VET_INSTRUCTIONS__C`: Text field with vet instruction. Unlikely to be useful, but keeping it anyway.
        * `LATEST_VET_VISIT_DATE__C`: Last date of vet visit.
        * `LIVED_WITH_PETS__C`: Indicator of whether the pet has lived with pets. Mostly blank, but non-blanks are "Yes", "No", and "Unknown".
        * `LIVED_WTIH_KIDS__C`: Indicator of whether the pet has lived with kids. Mostly blank, but non-blanks are "Yes", "No", and "Unknown".
        * `MICROCHIP_DATE_ISSUED__C`: Date microchipped.
        * `NAME`: Name of pet.
        * `ON_HOLD_UNTIL__C`:  Date. Mostly blank.
        * `OVERALL_HEALTH_NOTES__C`: Text of overall health notes.
        * `PRIMARY_BREED__C`
        * `PRIMARY_COLOR__C`
        * `RABIES_EXPIRATION__C`
        * `RESTRICTIONS_NOTES__C`: Text specifying adoption restriction criteria. Includes things like "adopt with Bambi" (another pet).
        * `RESTRICTIONS__C`: Categories of restrictions. Mostly blank but non-blanks are "No Dogs", "No Cats", "No Dogs;No Cats", "No Kids"
        * `SECONDARY_BREED__C`
        * `SECONDARY_COLOR__C`
        * `SIZE_AT_MATURITY__C`: Mostly blank. Non-blanks are "small", "medium", "large".
        * `SPAYED_NEUTERED__C`: Boolean spay/neuter indicator.
        * `SPECIES__C`: "Dog", "Cat", or "Other". Very few blanks.
        * `WEIGHT__C`: Mostly blank. Current weight in unknown units, likely pounds. May be able to fill blanks from _MedicalRecordExtract_.
* **AnimalDispositionExtract**: Each row appears to be a counselor's notes on an animal's disposition for each incoming/outgoing event. It is unique by intake/outgoing event and contains additional useful information, though many of the fields are redundant from _AnimalExtract_.
    - This table is unique by `ID`.
    - It can be joined to _AnimalExtract_ by `ANIMAL__C`.
    - Some potentially useful information includes:
        * `ADOPTED_BEFORE__C`: Boolean indicator of whether the animal was adopted before.
        * `INTAKE_NOTES__C`: Text notes from the intake event.
        * `INTAKE_REASON__C`: Categories of disposition. Mostly blank, but non-blanks include "Behavior Issues", "Escapes", "Moving", "Health of Animal", "Landlord Issues".
        * `TYPE_INTAKE_TRANSFER__C`: This appears to be categories of the event type. Values include "Adoption", "Euthanasia", "Clinic In", "Clinic Out", "Return to Owner". These may be useful, but they would have to be further categorized (e.g. incoming event or outgoing event.)
        * `DATE_OF_TRANSFER_INTAKE__C`: The date of the event.
        * `ANIMAL_STAGE__C`: Category with values that include "Ready to Adopt" and "Released". Could be combined with `DATE_OF_TRANSFER_INTAKE__C` to determine adoption availability date.
* There are a few medical-related tables that may be useful down the line, but they may not be worth focusing on at the outset. The tables are:
    - **MedicalEventExtract**: Records of each medical event.
    - **MedicalRecordExtract**: The medical record for each animal.
    - **MedicinesExtract**: Medicines for each pet. Appears to be unique by medicine, pet.
    - **VaccinationsExtract**: Vaccinations for each pet. Appears to be unique by vaccination, pet.
* **AssesmentsExtract**: This is the behavioral assessment for each pet. The fields contain text notes on behavioral tests like `EXPLORE_TEST__C` or on command responsiveness, like `FETCH_COMMAND__C`. This table could be used with NLP analysis to create additional features (flags or categories) or to create a standalone behavior prediction score.
