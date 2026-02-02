#ifndef CML_CORE_VIZ_SERVER_H
#define CML_CORE_VIZ_SERVER_H

#include <stdbool.h>

/**
 * @brief Initialize the embedded visualization server
 * @return 0 on success, negative on failure
 */
int viz_server_init(void);

/**
 * @brief Shutdown the embedded visualization server
 */
void viz_server_shutdown(void);

/**
 * @brief Broadcast a JSON message to all connected clients via SSE
 * @param event_type The type of event (e.g., "log", "graph")
 * @param json_data The JSON payload string
 */
void viz_server_broadcast(const char* event_type, const char* json_data);

#endif // CML_CORE_VIZ_SERVER_H
