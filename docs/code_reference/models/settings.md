# `Settings`

Redbox used the `pydantic_settings` library to manage settings. This library allows for settings to be defined in a type-safe way using Pydantic models. This is done by creating a `Settings` class that inherits from `BaseSettings` and defines the settings as class attributes.

::: redbox.models.settings.Settings

# OpenSearch Settings

We configure Opensearch via `OpenSearchSettings` in redbox-core/redbox/models/settings.py