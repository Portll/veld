import {
  getTasks,
  getProjects,
  createTask,
  completeTask,
  createProject as createTodoistProject,
  getProject,
  type TodoistTask,
  type TodoistProject,
} from "./todoist";
import {
  listTodos,
  listProjects,
  createTodo,
  createProject,
  completeTodo,
  updateTodo,
  type ShodhTodo,
  type ShodhProject,
} from "./shodh";

// Cache for project mappings
let todoistProjectCache: Map<string, TodoistProject> = new Map();
let shodhProjectCache: Map<string, ShodhProject> = new Map();

function todoistPriorityToShodh(priority: number): ShodhTodo["priority"] {
  switch (priority) {
    case 4: return "urgent";
    case 3: return "high";
    case 2: return "medium";
    default: return "none";
  }
}

function shodhPriorityToTodoist(priority: string): number {
  switch (priority) {
    case "urgent": return 4;
    case "high": return 3;
    case "medium": return 2;
    case "low": return 1;
    default: return 1;
  }
}

async function loadProjectCaches(): Promise<void> {
  // Load Todoist projects
  const todoistProjects = await getProjects();
  todoistProjectCache = new Map(todoistProjects.map(p => [p.id, p]));

  // Load shodh projects
  const shodhProjects = await listProjects();
  shodhProjectCache = new Map(shodhProjects.map(p => [p.name.toLowerCase(), p]));

  console.log(`📁 Loaded ${todoistProjects.length} Todoist projects, ${shodhProjects.length} shodh projects`);
}

async function getOrCreateShodhProject(todoistProjectId: string): Promise<string | undefined> {
  const todoistProject = todoistProjectCache.get(todoistProjectId);
  if (!todoistProject) {
    // Try to fetch it
    try {
      const proj = await getProject(todoistProjectId);
      todoistProjectCache.set(proj.id, proj);
    } catch {
      return undefined;
    }
  }

  const projectName = todoistProjectCache.get(todoistProjectId)?.name;
  if (!projectName) return undefined;

  // Check if shodh has this project
  const shodhProject = shodhProjectCache.get(projectName.toLowerCase());
  if (shodhProject) {
    return projectName;
  }

  // Create in shodh
  try {
    const newProject = await createProject(projectName);
    shodhProjectCache.set(projectName.toLowerCase(), newProject);
    console.log(`  📁 Created shodh project: ${projectName}`);
    return projectName;
  } catch (err: any) {
    console.error(`  ❌ Failed to create project ${projectName}: ${err.message}`);
    return undefined;
  }
}

async function getOrCreateTodoistProject(shodhProjectName: string): Promise<string | undefined> {
  // Check if Todoist has this project
  for (const [id, proj] of todoistProjectCache) {
    if (proj.name.toLowerCase() === shodhProjectName.toLowerCase()) {
      return id;
    }
  }

  // Create in Todoist
  try {
    const newProject = await createTodoistProject(shodhProjectName);
    todoistProjectCache.set(newProject.id, newProject);
    console.log(`  📁 Created Todoist project: ${shodhProjectName}`);
    return newProject.id;
  } catch (err: any) {
    console.error(`  ❌ Failed to create Todoist project ${shodhProjectName}: ${err.message}`);
    return undefined;
  }
}

async function syncTodoistToShodh(): Promise<void> {
  console.log("\n🔄 Todoist → shodh-memory...");

  const tasks = await getTasks();
  const todos = await listTodos(true);

  // Build map of external_id -> shodh todo
  const syncedByExternalId = new Map<string, ShodhTodo>();
  for (const todo of todos) {
    if (todo.external_id?.startsWith("todoist:")) {
      syncedByExternalId.set(todo.external_id, todo);
    }
  }

  let synced = 0;
  let skipped = 0;

  for (const task of tasks) {
    const externalId = `todoist:${task.id}`;

    // Skip if already synced
    if (syncedByExternalId.has(externalId)) {
      skipped++;
      continue;
    }

    // Get or create matching shodh project
    let projectName: string | undefined;
    if (task.project_id) {
      projectName = await getOrCreateShodhProject(task.project_id);
    }

    try {
      await createTodo({
        content: task.content,
        priority: todoistPriorityToShodh(task.priority),
        due_date: task.due?.date,
        tags: ["todoist", ...task.labels],
        external_id: externalId,
        project: projectName,
      });
      synced++;
      const projLabel = projectName ? ` [${projectName}]` : "";
      console.log(`  ✅ ${task.content.substring(0, 40)}${projLabel}`);
    } catch (err: any) {
      console.error(`  ❌ ${task.content.substring(0, 30)}: ${err.message}`);
    }
  }

  console.log(`  ${synced > 0 ? `Synced ${synced} tasks` : "No new tasks"} (${skipped} already synced)`);
}

async function syncShodhToTodoist(): Promise<void> {
  console.log("\n🔄 shodh-memory → Todoist...");

  const todos = await listTodos();
  const tasks = await getTasks();

  // Build set of Todoist task IDs for quick lookup
  const todoistTaskIds = new Set(tasks.map(t => t.id));

  // Only sync shodh todos that don't have an external_id (not from Todoist)
  const toSync = todos.filter(todo => !todo.external_id?.startsWith("todoist:"));

  let synced = 0;
  let skipped = 0;

  for (const todo of toSync) {
    // Skip if already has external_id pointing to valid Todoist task
    if (todo.external_id?.startsWith("todoist:")) {
      const todoistId = todo.external_id.replace("todoist:", "");
      if (todoistTaskIds.has(todoistId)) {
        skipped++;
        continue;
      }
    }

    // Get or create matching Todoist project
    let projectId: string | undefined;
    if (todo.project) {
      projectId = await getOrCreateTodoistProject(todo.project);
    }

    try {
      const task = await createTask({
        content: todo.content,
        priority: shodhPriorityToTodoist(todo.priority),
        due_string: todo.due_date ? new Date(todo.due_date).toLocaleDateString() : undefined,
        labels: todo.tags.filter(t => t !== "todoist"),
        project_id: projectId,
      });

      // Mark shodh todo as synced to Todoist
      await updateTodo(todo.id, { external_id: `todoist:${task.id}` });

      synced++;
      const projLabel = todo.project ? ` [${todo.project}]` : "";
      console.log(`  ✅ ${todo.content.substring(0, 40)}${projLabel}`);
    } catch (err: any) {
      console.error(`  ❌ ${todo.content.substring(0, 30)}: ${err.message}`);
    }
  }

  console.log(`  ${synced > 0 ? `Synced ${synced} todos` : "No new todos"} (${skipped} skipped)`);
}

async function syncCompletions(): Promise<void> {
  console.log("\n🔄 Syncing completions...");

  const todoistTasks = await getTasks();
  const todoistIds = new Set(todoistTasks.map(t => t.id));
  const shodhTodos = await listTodos(true);

  let completed = 0;

  for (const todo of shodhTodos) {
    if (!todo.external_id?.startsWith("todoist:")) continue;
    const todoistId = todo.external_id.replace("todoist:", "");

    // Task gone from Todoist = completed there, mark done in shodh
    if (!todoistIds.has(todoistId) && todo.status !== "done") {
      try {
        await completeTodo(todo.id);
        completed++;
        console.log(`  ✅ Done in shodh: ${todo.content.substring(0, 40)}`);
      } catch (err) {}
    }

    // Done in shodh but still open in Todoist = complete in Todoist
    if (todo.status === "done" && todoistIds.has(todoistId)) {
      try {
        await completeTask(todoistId);
        completed++;
        console.log(`  ✅ Done in Todoist: ${todo.content.substring(0, 40)}`);
      } catch (err) {}
    }
  }

  console.log(`  ${completed > 0 ? `${completed} completions synced` : "No completions"}`);
}

export async function runSync(): Promise<void> {
  console.log("\n" + "─".repeat(50));
  console.log("🔄 Todoist ↔ shodh-memory (with project sync)");
  console.log("─".repeat(50));

  try {
    await loadProjectCaches();
    await syncTodoistToShodh();
    await syncShodhToTodoist();
    await syncCompletions();
    console.log("\n✅ Sync complete");
  } catch (err) {
    console.error("\n❌ Sync failed:", err);
  }
}

if (import.meta.main) {
  runSync();
}
