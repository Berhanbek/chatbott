import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

export function FaqSection() {
  return (
    <Card className="retro-border bg-secondary/50">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm terminal-text">Frequently Asked Questions</CardTitle>
      </CardHeader>
      <CardContent className="text-xs terminal-text">
        <Accordion type="single" collapsible className="w-full">
          <AccordionItem value="item-1">
            <AccordionTrigger className="text-xs">What can SiSChatBot help me with?</AccordionTrigger>
            <AccordionContent className="text-xs">
              SiSChatBot can help with course-related questions, explanations of concepts, assignment help, department
              information, faculty details, and calculating your GPA.
            </AccordionContent>
          </AccordionItem>

          <AccordionItem value="item-2">
            <AccordionTrigger className="text-xs">How do I register for courses at AAU?</AccordionTrigger>
            <AccordionContent className="text-xs">
              Course registration is done through the university portal. Log in with your student ID, navigate to
              "Course Registration" and follow the prompts to select your courses for the upcoming semester.
            </AccordionContent>
          </AccordionItem>

          <AccordionItem value="item-3">
            <AccordionTrigger className="text-xs">What are the graduation requirements for BIS?</AccordionTrigger>
            <AccordionContent className="text-xs">
              To graduate with a Bachelor of Science in Information Systems, you need to complete all required courses
              in the 4-year program, maintain a satisfactory GPA, and complete your final year project. Check with your
              academic advisor for specific requirements for your batch.
            </AccordionContent>
          </AccordionItem>

          <AccordionItem value="item-4">
            <AccordionTrigger className="text-xs">How do I contact my professors?</AccordionTrigger>
            <AccordionContent className="text-xs">
              You can contact professors through their university email or during their office hours. Contact
              information for all faculty members is available in the Faculty & Staff section of this app or on the
              department website.
            </AccordionContent>
          </AccordionItem>

          <AccordionItem value="item-5">
            <AccordionTrigger className="text-xs">Where can I find academic resources?</AccordionTrigger>
            <AccordionContent className="text-xs">
              Academic resources are available at the university library, learning center, and online through the
              university portal. You can also access lecture notes, PDFs, past exams, and assignments via the SIS
              Resource Telegram Bot @SISResourcesBot.
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </CardContent>
    </Card>
  )
}
