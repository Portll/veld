import { config } from "./config";

export interface TodoistTask {
  id: string;
  content: string;
  description: string;
  is_completed: boolean;
  priority: number; // 1 = normal, 4 = urgent
  due?: {
    date: string;
    string: string;
  };
  labels: string[];
  project_id: string;
}

export interface TodoistProject {
  id: string;
  name: string;
}

async function apiCall<T>(
  endpoint: string,
  method: string = "GET",
  body?: object
): Promise<T> {
  const url = `${config.todoist.apiUrl}${endpoint}`;

  const res = await fetch(url, {
    method,
    headers: {
      "Authorization": `Bearer ${config.todoist.apiToken}`,
      "Content-Type": "application/json",
    },
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Todoist API error: ${res.status} ${text}`);
  }

  if (res.status === 204) return {} as T;
  return res.json();
}

export async function getTasks(): Promise<TodoistTask[]> {
  return apiCall<TodoistTask[]>("/tasks");
}

export async function getCompletedTasks(since?: string): Promise<any[]> {
  const params = new URLSearchParams();
  if (since) params.set("since", since);
  return apiCall<any[]>(`/tasks/completed?${params}`);
}

export async function createTask(task: {
  content: string;
  description?: string;
  priority?: number;
  due_string?: string;
  labels?: string[];
  project_id?: string;
}): Promise<TodoistTask> {
  return apiCall<TodoistTask>("/tasks", "POST", task);
}

export async function getProject(projectId: string): Promise<TodoistProject> {
  return apiCall<TodoistProject>(`/projects/${projectId}`);
}

export async function updateTask(
  taskId: string,
  updates: Partial<TodoistTask>
): Promise<TodoistTask> {
  return apiCall<TodoistTask>(`/tasks/${taskId}`, "POST", updates);
}

export async function completeTask(taskId: string): Promise<void> {
  await apiCall(`/tasks/${taskId}/close`, "POST");
}

export async function reopenTask(taskId: string): Promise<void> {
  await apiCall(`/tasks/${taskId}/reopen`, "POST");
}

export async function deleteTask(taskId: string): Promise<void> {
  await apiCall(`/tasks/${taskId}`, "DELETE");
}

export async function getProjects(): Promise<TodoistProject[]> {
  return apiCall<TodoistProject[]>("/projects");
}

export async function createProject(name: string): Promise<TodoistProject> {
  return apiCall<TodoistProject>("/projects", "POST", { name });
}
