/** Entrypoint for frontend js **/

/* Packages */

// HTMX
import 'htmx.org';
import htmx from 'htmx.org';
htmx.config.includeIndicatorStyles = false;

// GOVUK-FRONTEND
import { initAll } from "govuk-frontend";
initAll();

// I.AI DESIGN SYSTEM
import "../../node_modules/i.ai-design-system/dist/iai-design-system.js";

// REDBOX DESIGN SYSTEM
import "../redbox_design_system/rbds";


/* Application logic */

import "./csrftoken.js";
import './menu.js';
import "./trusted-types.js";
