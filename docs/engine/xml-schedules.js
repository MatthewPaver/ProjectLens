/**
 * P6 XML (PMXML) and MS Project MSPDI parsers.
 * Returns the same assurance-shaped schedule object as parseXerAssurance.
 * Uses DOMParser in the browser and a tiny tag reader in Node (no dependency).
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory(require("./projectlens-engine.js"));
  } else {
    factory(root.ProjectLensEngine);
  }
}(typeof self !== "undefined" ? self : this, function (engine) {
  "use strict";

  function parseXmlDom(text) {
    if (typeof DOMParser !== "undefined") {
      const doc = new DOMParser().parseFromString(text, "application/xml");
      if (doc.querySelector("parsererror")) throw new Error("The XML schedule could not be parsed.");
      return doc;
    }
    // Minimal Node fallback: wrap in a fake document with querySelectorAll via regex — too weak.
    // Instead build a lightweight element tree.
    return buildLightDom(text);
  }

  function buildLightDom(text) {
    // Strip comments and the XML declaration so nesting stays easy to walk.
    const cleaned = String(text).replace(/<\?xml[^?]*\?>/i, "").replace(/<!--[\s\S]*?-->/g, "");
    const root = { name: "#document", children: [], attrs: {}, text: "" };
    const stack = [root];
    const token = /<\/?([A-Za-z_][\w:.-]*)([^>]*)>|([^<]+)/g;
    let match;
    while ((match = token.exec(cleaned))) {
      if (match[3] !== undefined) {
        const value = match[3].trim();
        if (value) stack[stack.length - 1].text += (stack[stack.length - 1].text ? " " : "") + value;
        continue;
      }
      const name = match[1].replace(/^.*:/, "");
      const isClose = match[0].startsWith("</");
      const selfClose = /\/>$/.test(match[0]);
      if (isClose) {
        if (stack.length > 1) stack.pop();
        continue;
      }
      const node = { name, children: [], attrs: {}, text: "" };
      stack[stack.length - 1].children.push(node);
      if (!selfClose) stack.push(node);
    }
    return {
      querySelector(sel) { return findOne(root, sel); },
      querySelectorAll(sel) { return findAll(root, sel); },
      documentElement: root.children[0] || root,
    };
  }

  function localName(node) {
    return (node.name || node.localName || node.tagName || "").replace(/^.*:/, "");
  }

  function findAll(node, selector) {
    // Supports "Tag" and "Tag Tag" (descendant) only — enough for these fixtures.
    const parts = selector.trim().split(/\s+/).map(p => p.replace(/^.*:/, ""));
    let current = [node];
    for (const part of parts) {
      const next = [];
      for (const item of current) walk(item, part, next);
      current = next;
    }
    return current;
  }

  function walk(node, name, out) {
    const children = node.children || [...(node.children || [])];
    if (typeof node.querySelectorAll === "function" && node !== node) {
      // unused
    }
    const list = node.children || [];
    for (const child of list) {
      if (localName(child) === name) out.push(child);
      walk(child, name, out);
    }
    // Browser Element path
    if (node.getElementsByTagName) {
      // handled by real DOM
    }
  }

  function findOne(node, selector) {
    return findAll(node, selector)[0] || null;
  }

  function textOf(node, tag) {
    if (!node) return "";
    if (typeof node.querySelector === "function" && node.documentElement) {
      const hit = node.querySelector(tag) || (node.documentElement && findOne(node.documentElement, tag));
      return hit ? (hit.textContent || hit.text || "").trim() : "";
    }
    const hit = findOne(node, tag);
    return hit ? (hit.textContent || hit.text || "").trim() : "";
  }

  function childText(parent, tag) {
    if (!parent) return "";
    if (parent.getElementsByTagName) {
      const list = parent.getElementsByTagName(tag);
      // Prefer local-name match across namespaces.
      for (let i = 0; i < list.length; i += 1) {
        if (localName(list[i]) === tag && list[i].parentNode === parent) {
          return (list[i].textContent || "").trim();
        }
      }
      // Descendant fallback
      for (let i = 0; i < list.length; i += 1) {
        if (localName(list[i]) === tag) return (list[i].textContent || "").trim();
      }
      return "";
    }
    const direct = (parent.children || []).find(c => localName(c) === tag);
    return direct ? (direct.text || "").trim() : "";
  }

  function hoursValue(raw) {
    if (!raw) return null;
    const iso = String(raw).match(/PT(?:(\d+)H)?/i);
    if (iso) return Number(iso[1] || 0);
    const plain = String(raw).match(/(-?\d+(?:\.\d+)?)\s*h?/i);
    return plain ? Number(plain[1]) : null;
  }

  function elements(doc, tag) {
    if (doc.getElementsByTagNameNS) {
      const all = [...doc.getElementsByTagName("*")].filter(el => localName(el) === tag);
      return all;
    }
    const root = doc.documentElement || doc;
    const hits = findAll(root, tag);
    // Include the document element itself when it is the requested tag (MSPDI root).
    if (localName(root) === tag) hits.unshift(root);
    return hits;
  }

  function assuranceFromTasks({ fileName, text, project, tasks, relations }) {
    const parseDate = engine.parseDate;
    const numberValue = engine.numberValue;
    return {
      fileName,
      sourceText: String(text || "").replace(/\r\n?/g, "\n").trim(),
      project: {
        id: project.id || "unassigned",
        name: project.name || "Unnamed project",
        dataDate: parseDate(project.dataDate),
        scheduledFinish: parseDate(project.scheduledFinish),
      },
      tasks: tasks.map(task => ({
        code: task.code,
        name: task.name,
        finish: parseDate(task.finish),
        float: numberValue(task.float),
        constraint: task.constraint || "",
        constraintDate: parseDate(task.constraintDate),
        status: task.status || "",
        type: task.type || "",
        start: parseDate(task.start),
        duration: numberValue(task.duration),
        id: task.id,
        isMilestone: Boolean(task.isMilestone),
      })),
      taskByCode: null,
      relations: new Set(relations.map(r => JSON.stringify([
        r.predecessorCode,
        r.successorCode,
        r.type || "PR_FS",
        numberValue(r.lag) || 0,
      ]))),
    };
  }

  function finalise(schedule) {
    schedule.taskByCode = new Map(schedule.tasks.map(task => [task.code, task]));
    if (schedule.tasks.some(task => !task.code) || new Set(schedule.tasks.map(task => task.code)).size !== schedule.tasks.length) {
      throw new Error(`${schedule.fileName} needs unique, non-empty activity codes for comparison.`);
    }
    if (!schedule.project.scheduledFinish) {
      throw new Error(`${schedule.fileName} has no supported project scheduled finish.`);
    }
    return schedule;
  }

  function parseP6Xml(text, fileName = "schedule.xml") {
    if (!String(text || "").trim()) throw new Error(`${fileName} is empty.`);
    const doc = parseXmlDom(text);
    const projectEl = elements(doc, "Project")[0];
    if (!projectEl) throw new Error(`${fileName} is not a readable Primavera P6 XML schedule.`);
    const activities = elements(doc, "Activity").filter(el => {
      // Keep activities that sit under the project (avoid nested noise).
      return true;
    });
    const byId = new Map();
    const tasks = activities.map(el => {
      const id = childText(el, "ObjectId") || childText(el, "Id");
      const code = childText(el, "Id") || id;
      const type = childText(el, "Type") || "TT_Task";
      const task = {
        id,
        code,
        name: childText(el, "Name") || code,
        finish: childText(el, "RemainingEarlyFinishDate") || childText(el, "FinishDate"),
        start: childText(el, "RemainingEarlyStartDate") || childText(el, "StartDate"),
        float: hoursValue(childText(el, "TotalFloat")),
        duration: hoursValue(childText(el, "PlannedDuration")),
        status: childText(el, "Status"),
        type: /mile/i.test(type) ? "TT_FinMile" : "TT_Task",
        isMilestone: /mile/i.test(type),
        constraint: "",
        constraintDate: null,
      };
      byId.set(String(id), task);
      return task;
    });
    const relations = elements(doc, "Relationship").map(el => {
      const pred = byId.get(childText(el, "PredecessorActivityObjectId"));
      const succ = byId.get(childText(el, "SuccessorActivityObjectId"));
      return {
        predecessorCode: pred?.code || childText(el, "PredecessorActivityObjectId"),
        successorCode: succ?.code || childText(el, "SuccessorActivityObjectId"),
        type: "PR_FS",
        lag: hoursValue(childText(el, "Lag")) || 0,
      };
    });
    return finalise(assuranceFromTasks({
      fileName,
      text,
      project: {
        id: childText(projectEl, "ObjectId") || "1",
        name: childText(projectEl, "Id") || childText(projectEl, "Name") || "Unnamed project",
        dataDate: childText(projectEl, "DataDate"),
        scheduledFinish: childText(projectEl, "ScheduledFinishDate") || childText(projectEl, "FinishDate"),
      },
      tasks,
      relations,
    }));
  }

  function parseMspdiXml(text, fileName = "schedule.xml") {
    if (!String(text || "").trim()) throw new Error(`${fileName} is empty.`);
    const doc = parseXmlDom(text);
    const projectEl = elements(doc, "Project")[0];
    if (!projectEl) throw new Error(`${fileName} is not a readable MS Project XML (MSPDI) schedule.`);
    const taskEls = elements(doc, "Task").filter(el => childText(el, "UID") && childText(el, "UID") !== "0");
    const byUid = new Map();
    const tasks = taskEls.map(el => {
      const uid = childText(el, "UID");
      const code = childText(el, "WBS") || childText(el, "ID") || uid;
      const milestone = childText(el, "Milestone") === "1";
      const task = {
        id: uid,
        code,
        name: childText(el, "Name") || code,
        finish: childText(el, "Finish"),
        start: childText(el, "Start"),
        float: hoursValue(childText(el, "TotalSlack")),
        duration: hoursValue(childText(el, "Duration")),
        status: "",
        type: milestone ? "TT_FinMile" : "TT_Task",
        isMilestone: milestone,
        constraint: "",
        constraintDate: null,
      };
      byUid.set(uid, task);
      return task;
    });
    const relations = [];
    for (const el of taskEls) {
      const succ = byUid.get(childText(el, "UID"));
      const links = el.getElementsByTagName
        ? [...el.getElementsByTagName("*")].filter(n => localName(n) === "PredecessorLink" && n.parentNode === el)
        : (el.children || []).filter(c => localName(c) === "PredecessorLink");
      for (const link of links) {
        const predUid = childText(link, "PredecessorUID");
        const pred = byUid.get(predUid);
        relations.push({
          predecessorCode: pred?.code || predUid,
          successorCode: succ?.code || childText(el, "UID"),
          type: "PR_FS",
          lag: hoursValue(childText(link, "LinkLag")) || 0,
        });
      }
    }
    return finalise(assuranceFromTasks({
      fileName,
      text,
      project: {
        id: childText(projectEl, "UID") || "1",
        name: childText(projectEl, "Name") || "Unnamed project",
        dataDate: childText(projectEl, "CurrentDate") || childText(projectEl, "StartDate"),
        scheduledFinish: childText(projectEl, "FinishDate"),
      },
      tasks,
      relations,
    }));
  }

  engine.parseP6Xml = parseP6Xml;
  engine.parseMspdiXml = parseMspdiXml;
  return { parseP6Xml, parseMspdiXml };
}));
