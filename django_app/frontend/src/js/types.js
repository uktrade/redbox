// @ts-check

/**
 * @typedef {Object} SelectedDocument
 *
 * @property {string} id
 * @property {string} name
*/


/**
 * @typedef {Object} MessageCreatedResponse
 *
 * @property {string} chat_message_id
 * @property {string} chat_message_role
 * @property {string} html
 */


/**
 * @typedef {Object} MessageUpdateResponse
 *
 * @property {string} chat_message_id
 * @property {string} html
 * @property {string} sr_text
 */


/**
 * @typedef {Object} MessageCompleteResponse
 *
 * @property {string} chat_message_id
 * @property {string} html
 * @property {string} title
 * @property {string} session_id
 */


/**
 * @typedef {Object} MessageActivityResponse
 *
 * @property {string} chat_message_id - ChatMessage ID
 * @property {string} activity_event_message - ActivityEvent message
 */


/**
 * @typedef {Object} StreamOptions
 *
 * @property {string} text
 * @property {string[]} documents - File UUIDs
 * @property {string} llm
 * @property {string | undefined} toolId
 * @property {string[]} activities
 */
