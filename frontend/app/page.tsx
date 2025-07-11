// app/page.tsx (or wherever your "/" route component is defined)
import { redirect } from "next/navigation";

export default function Home() {
  redirect("/dashboard");
}
