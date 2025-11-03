/** Entrypoint for frontend js **/

/* Packages */

// HTMX
import 'htmx.org';
import htmx from 'htmx.org';
htmx.config.includeIndicatorStyles = false;

// GOVUK-FRONTEND
import { initAll } from "govuk-frontend";
initAll();

// REDBOX DESIGN SYSTEM
import "../redbox_design_system/rbds";


/* Application logic */

import "./csrftoken.js";
import './menu.js';
import "./trusted-types.js";
