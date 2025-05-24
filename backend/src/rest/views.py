from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status


class HelloWorldView(APIView):
    def get(self, _request: Request) -> Response:
        return Response({'message': 'Hello World'}, status=status.HTTP_200_OK)
