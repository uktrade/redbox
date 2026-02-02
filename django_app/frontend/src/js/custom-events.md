# Custom Events

Custom events are used for communication between web components. Here is a list of all available events.

| Event                | Pages                 | Data                                                    | Description                                                                               |
| -------------------- | --------------------- | ------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| chat-response-start  | /chats                | (none)                                                  | When the streaming connection is opened                                                   |
| chat-response-end    | /chats                | title: string<br/>session_id: string                    | When the stream "end" event is sent from the server                                       |
| doc-complete         | /chats<br/>/documents | file-status: HTMLElement                                | When a document status changes to "complete"                                              |
| selected-docs-change | /chats                | {id: string, name: string}[]                            | When a user selects or deselects a document                                               |
| start-streaming      | /chats                | (none)                                                  | When a user submits a message                                                             |
| stop-streaming       | /chats                | (none)                                                  | When a user presses the stop-streaming button, or an unexpected disconnection has occured |
| chat-title-change    | /chats                | title: string<br/>session_id: string<br/>sender: string | When the chat title is changed by the user                                                |
| file-upload-processed    | /chats                | (none) | When a individual file has finished processing processing                                                |
| file-uploads-processed    | /chats                | (none) | When all file uploads have finished processing                                                |
| file-uploads-removed    | /chats                | (none) | When all file uploads have been removed                                                |
|doc-selection-change    | /chats                | id: string<br/>name: string<br/>checked: bool | When a document has been selected/deselected in the side panel                                                |
