import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export function ProgramInfo() {
  return (
    <Card className="retro-border bg-secondary/50">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm terminal-text">Program Information</CardTitle>
      </CardHeader>
      <CardContent className="text-xs space-y-4 terminal-text">
        <div>
          <h3 className="font-bold mb-1 text-primary">Bachelor of Science in Information Systems (BIS)</h3>
          <p>A dynamic 4-year undergraduate program blending computing, management, and systems thinking.</p>
          <p className="mt-1">
            Prepares students for careers in software development, database design, systems analysis, IT consulting, and
            more.
          </p>
        </div>

        <div>
          <h3 className="font-bold mb-1 text-primary">Core Courses</h3>
          <ul className="list-disc pl-4 space-y-1">
            <li>Programming (C++, Java, Web Development)</li>
            <li>Database Systems (Advanced DB, SQL)</li>
            <li>System Analysis & Design</li>
            <li>Cybersecurity & IT Infrastructure</li>
            <li>Data Structures & Algorithms</li>
            <li>Project Management</li>
            <li>Enterprise Systems</li>
          </ul>
          <p className="mt-2 italic">Full course outlines are available upon request.</p>
        </div>

        <div>
          <h3 className="font-bold mb-1 text-primary">Program Structure</h3>
          <div className="space-y-2">
            <div>
              <h4 className="font-semibold text-accent">Year 1</h4>
              <p>Foundation courses in computing, mathematics, and general education.</p>
            </div>
            <div>
              <h4 className="font-semibold text-accent">Year 2</h4>
              <p>Core programming, database fundamentals, and system design principles.</p>
            </div>
            <div>
              <h4 className="font-semibold text-accent">Year 3</h4>
              <p>Advanced topics in information systems, specialized electives, and project work.</p>
            </div>
            <div>
              <h4 className="font-semibold text-accent">Year 4</h4>
              <p>Capstone projects, professional practice, and specialized concentration areas.</p>
            </div>
          </div>
        </div>

        <div>
          <h3 className="font-bold mb-1 text-primary">Career Opportunities</h3>
          <ul className="list-disc pl-4 space-y-1">
            <li>Software Developer</li>
            <li>Database Administrator</li>
            <li>Systems Analyst</li>
            <li>IT Consultant</li>
            <li>Business Analyst</li>
            <li>Project Manager</li>
            <li>Information Systems Manager</li>
          </ul>
        </div>
      </CardContent>
    </Card>
  )
}
