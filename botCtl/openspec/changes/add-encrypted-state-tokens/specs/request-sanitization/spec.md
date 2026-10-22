# Spec Delta

## Purpose

Protects the bot service from injection-prone user input by validating
message text at the API boundary before any processing, while leaving
normal language input (including umlauts) untouched.

## ADDED Requirements

### Requirement: Message text must be plain text

The system SHALL accept only plain text as `message.text`: a non-empty
string containing no HTML tags, no injection-prone characters, and no
control characters beyond common whitespace. Violating input MUST be
rejected with HTTP 400 before intent detection or any pipeline
processing runs, and the rejection MUST NOT alter stored history or
conversation state.

#### Scenario: HTML tags are rejected

- **WHEN** the message text contains an HTML tag such as `<script>`
- **THEN** the backend responds with HTTP 400 and does not run intent
  detection on the input

#### Scenario: Injection-prone characters are rejected

- **WHEN** the message text contains any of the characters backtick,
  opening or closing brace, pipe, dollar, semicolon, ampersand, or
  angle bracket
- **THEN** the backend responds with HTTP 400

#### Scenario: Control characters are rejected

- **WHEN** the message text contains a C0 control character other than
  tab, newline, or carriage return
- **THEN** the backend responds with HTTP 400

#### Scenario: Normal German text is accepted

- **WHEN** the message text is an ordinary German question containing
  umlauts, such as `"was macht das reh in der aue?"`
- **THEN** the backend processes the request normally with status 200

#### Scenario: Empty message text is rejected

- **WHEN** the message text is missing, empty, or only whitespace
- **THEN** the backend responds with HTTP 400
