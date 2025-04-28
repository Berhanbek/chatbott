import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export function DepartmentInfo() {
  return (
    <Card className="retro-border bg-secondary/50">
      <CardHeader className="pb-2">
        {/**/}
        <img
          src="c:\Users\Berhan\Desktop\chattyb\frontend\aau.PNG" // Replace with your image path
          alt="Department Overview"
          className="w-full h-40 object-cover rounded-md mb-2"
        />
        <CardTitle className="text-sm terminal-text">Department Information</CardTitle>
      </CardHeader>
      <CardContent className="text-xs space-y-4 terminal-text">
        <div>
          <h3 className="font-bold mb-1 text-primary">Department Overview</h3>
          <p>
            The School of Information Science is a vital part of the College of Natural and Computational Sciences at
            AAU. It is dedicated to empowering students with both technical expertise and managerial insight across
            diverse areas in Information Systems.
          </p>
        </div>

        <div>
          <h3 className="font-bold mb-1 text-primary">Location & Contact Details</h3>
          <p>Location: Eshetu Chole Building, FBE Campus, 1st–6th Floors</p>
          <p>Phone: +251 11 122 9191</p>
          <p>Email: info@aau.edu.et</p>
        </div>

        <div>
          <h3 className="font-bold mb-1 text-primary">Coordinators</h3>
          <p>Ato Betsegaw Dereje</p>
          <p>Email: betsegaw.dereje@aau.edu.et</p>
          <p>Office: Room 423</p>
          <p className="mt-2">Dr. Tibebe Beshah</p>
          <p>Email: tibebe.beshah@aau.edu.et</p>
          <p>Office: Room 422</p>
        </div>

        <div>
          <h3 className="font-bold mb-1 text-primary">Events & Student Community</h3>
          <p>SIS is more than academics — it's a vibrant student community! The department organizes:</p>
          <ul className="list-disc pl-4 mt-1 space-y-1">
            <li>Hackathons</li>
            <li>IS Talks & Seminars</li>
            <li>Game Fests & Innovation Challenges</li>
            <li>Graduation Exhibitions</li>
            <li>IS Day & Community Meetups</li>
          </ul>
        </div>

        <div>
          <h3 className="font-bold mb-1 text-primary">Student Resources</h3>
          <p>
            Resource Bot: Access lecture notes, PDFs, past exams, and assignments via the SIS Resource Telegram Bot
            @SISResourcesBot
          </p>
          <p className="mt-1">
            SiSChatBot (TheISSeer): Ask questions about the department, instructors, course concepts, and more — powered
            by Gemini AI.
          </p>
        </div>
      </CardContent>
    </Card>
  )
}
